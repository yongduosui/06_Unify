import os
import random
import argparse

import torch
import torch.nn as nn
import numpy as np

import net as net
from utils import load_data
from sklearn.metrics import f1_score
import pdb
import pruning


def run(args, seed):

    pruning.setup_seed(seed)
    adj, features, labels, idx_train, idx_val, idx_test = load_data(args['dataset'])
    adj = adj.to_dense()

    node_num = features.size()[0]
    class_num = labels.numpy().max() + 1

    adj = adj.cuda()
    features = features.cuda()
    labels = labels.cuda()

    loss_func = nn.CrossEntropyLoss()
    early_stopping = 10

    net_gcn = net.net_gcn(embedding_dim=args['embedding_dim'], adj=adj)
    pruning.add_mask(net_gcn)

    net_gcn = net_gcn.cuda()
    optimizer = torch.optim.Adam(net_gcn.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    loss_val = []
    for epoch in range(1000):

        pruning.plot_mask_distribution(net_gcn, epoch, "mask_distribution")
        optimizer.zero_grad()
        output = net_gcn(features, adj)
        loss = loss_func(output[idx_train], labels[idx_train])
        # print('epoch', epoch, 'loss', loss_train.data)
        loss.backward()
        pruning.subgradient_update_mask(net_gcn, args) # l1 norm
        optimizer.step()
        # validation
        with torch.no_grad():
            output = net_gcn(features, adj, val_test=True)
            loss_val.append(loss_func(output[idx_val], labels[idx_val]).cpu().numpy())
            # print('val acc', f1_score(labels[idx_val].cpu().numpy(), output[idx_val].cpu().numpy().argmax(axis=1), average='micro'))
        # early stopping
        if epoch > early_stopping and loss_val[-1] > np.mean(loss_val[-(early_stopping+1):-1]):
            break
    pdb.set_trace()
    # adj_mask_tensor, weight_mask_tensor = pruning.get_mask_distribution(net_gcn)
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
    parser.add_argument('--s1', type=float, default=0.0001,help='scale sparse rate (default: 0.0001)')
    parser.add_argument('--s2', type=float, default=0.0001,help='scale sparse rate (default: 0.0001)')
    return parser




if __name__ == "__main__":

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    parser = parser_loader()
    args = vars(parser.parse_args())
    print(args)

    seed_time = 20
    acc_val = np.zeros(seed_time)
    acc_test = np.zeros(seed_time)
    for seed in range(seed_time):
        acc_val[seed], acc_test[seed], epoch = run(args, seed)
        print("Seed:[{}], Val:[{:.2f}], Test:[{:.2f}] Stop at epoch:[{}]".format(seed, acc_val[seed] * 100, acc_test[seed] * 100, epoch))

    print('Finish !')
    print('Val  mean : [{:.4f}]  std : [{:.4f}]'.format(acc_val.mean() * 100, acc_val.std() * 100))
    print('Test mean : [{:.4f}]  std : [{:.4f}]'.format(acc_test.mean() * 100, acc_test.std() * 100))

