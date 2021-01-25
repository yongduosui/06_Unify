import os
import random
import argparse

import torch
import torch.nn as nn
import numpy as np

import net as net
from utils import load_data
from sklearn.metrics import f1_score
import utils
import pdb
import pruning

def run(args, index, wei_percent, seed):

    adj = np.load("./admm_{}/adj_{}.npy".format(args['dataset'], index))
    adj = utils.normalize_adj(adj)
    adj = utils.sparse_mx_to_torch_sparse_tensor(adj)

    setup_seed(seed)

    _, features, labels, idx_train, idx_val, idx_test = load_data(args['dataset'])
    adj = adj.to_dense()
    
    node_num = features.size()[0]
    class_num = labels.numpy().max() + 1

    adj = adj.cuda()
    features = features.cuda()
    labels = labels.cuda()

    loss_func = nn.CrossEntropyLoss()

    net_gcn = net.net_gcn_baseline(embedding_dim=args['embedding_dim'])
    net_gcn = net_gcn.cuda()

    pruning.oneshot_weight_magnitude_pruning(net_gcn, wei_percent)

    for name, param in net_gcn.named_parameters():
        if 'mask' in name:
            param.requires_grad = False

    optimizer = torch.optim.Adam(net_gcn.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    acc_test = 0.0
    best_val_acc = {'val_acc': 0, 'epoch' : 0, 'test_acc':0}

    for epoch in range(200):

        optimizer.zero_grad()
        output = net_gcn(features, adj)
        loss = loss_func(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            output = net_gcn(features, adj, val_test=True)
            acc_val = f1_score(labels[idx_val].cpu().numpy(), output[idx_val].cpu().numpy().argmax(axis=1), average='micro')
            acc_test = f1_score(labels[idx_test].cpu().numpy(), output[idx_test].cpu().numpy().argmax(axis=1), average='micro')
            if acc_val > best_val_acc['val_acc']:
                best_val_acc['test_acc'] = acc_test
                best_val_acc['val_acc'] = acc_val
                best_val_acc['epoch'] = epoch

        print("(ADMM Fix Mask) Epoch:[{}] Val:[{:.2f}] Test:[{:.2f}] | Final Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}]"
                 .format(epoch, acc_val * 100, acc_test * 100, 
                                best_val_acc['val_acc'] * 100, 
                                best_val_acc['test_acc'] * 100, 
                                best_val_acc['epoch']))

    return best_val_acc['val_acc'], best_val_acc['test_acc'], best_val_acc['epoch']
    

    # test
    with torch.no_grad():
        output = net_gcn(features, adj, val_test=True)
        acc_val = f1_score(labels[idx_val].cpu().numpy(), output[idx_val].cpu().numpy().argmax(axis=1), average='micro')
        acc_test = f1_score(labels[idx_test].cpu().numpy(), output[idx_test].cpu().numpy().argmax(axis=1), average='micro')

    return acc_val, acc_test, epoch


def parser_loader():
    parser = argparse.ArgumentParser(description='Self-Supervised GCN')
    parser.add_argument('--dataset', type=str, default='citeseer')
    parser.add_argument('--embedding-dim', nargs='+', type=int, default=[3703,16,6])
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--pruning_percent_wei', type=float, default=0.2)
    parser.add_argument('--pruning_percent_adj', type=float, default=0.05)
    return parser


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = parser_loader()
    args = vars(parser.parse_args())
    print(args)
    seed = 3333

    percent_list = [(1 - (1 - args['pruning_percent_adj']) ** (i + 1), 1 - (1 - args['pruning_percent_wei']) ** (i + 1)) for i in range(20)]
    for index, (adj_percent, wei_percent) in enumerate(percent_list):

        acc_val, acc_test, epoch = run(args, index + 1, wei_percent, seed)
        print("ADMM Val:[{:.2f}], Test:[{:.2f}] at epoch:[{}] | INDEX:[{}] WEI:[{:.2f}]"
        .format(acc_val * 100, acc_test * 100, epoch, index, (1 - wei_percent) * 100))