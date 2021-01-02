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
import copy

import warnings
warnings.filterwarnings('ignore')

def run_pruning_acc(args, seed):

    pruning.setup_seed(seed)
    adj, features, labels, idx_train, idx_val, idx_test = load_data(args['dataset'])
    adj = adj.to_dense()

    node_num = features.size()[0]
    class_num = labels.numpy().max() + 1

    adj = adj.cuda()
    features = features.cuda()
    labels = labels.cuda()
    loss_func = nn.CrossEntropyLoss()

    net_gcn = net.net_gcn(embedding_dim=args['embedding_dim'], adj=adj)
    pruning.add_mask(net_gcn)
    net_gcn = net_gcn.cuda()
    optimizer = torch.optim.Adam(net_gcn.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    acc_test = 0.0
    supervise_loss = []
    adj_mask_loss = []
    weight_mask_loss = []
    test_acc_list = []
    for epoch in range(args['total_epoch']):
        # pruning.plot_mask_distribution(net_gcn, epoch, acc_test, "mask_distribution")
        optimizer.zero_grad()
        output = net_gcn(features, adj)
        loss = loss_func(output[idx_train], labels[idx_train])
        loss.backward()
        pruning.subgradient_update_mask(net_gcn, args) # l1 norm
        optimizer.step()

        supervise_loss.append(loss.detach().cpu().item())
        adj_mask_loss.append(net_gcn.adj_mask.data.abs().sum().detach().cpu().numpy().item())
        weight_mask_loss.append(net_gcn.net_layer[0].weight_mask_weight.data.abs().sum().detach().cpu().numpy() 
                              + net_gcn.net_layer[1].weight_mask_weight.data.abs().sum().detach().cpu().numpy())

        with torch.no_grad():
            output = net_gcn(features, adj, val_test=True)
            acc_test = f1_score(labels[idx_test].cpu().numpy(), output[idx_test].cpu().numpy().argmax(axis=1), average='micro')
            test_acc_list.append(acc_test)
            print("(Pruning Acc) Epoch:[{}] Test Acc[{:.2f}]".format(epoch, acc_test * 100))
    
    np.savez("./cora_mask_info", supervise_loss=supervise_loss, adj_mask_loss=adj_mask_loss, weight_mask_loss=weight_mask_loss, test_acc_list=test_acc_list)
    return acc_test, epoch


def parser_loader():
    parser = argparse.ArgumentParser(description='Self-Supervised GCN')
    ###### Unify pruning settings #######
    parser.add_argument('--s1', type=float, default=0.0001,help='scale sparse rate (default: 0.0001)')
    parser.add_argument('--s2', type=float, default=0.0001,help='scale sparse rate (default: 0.0001)')
    parser.add_argument('--total_epoch', type=int, default=130)
    parser.add_argument('--pruning_percent', type=float, default=0.1)

    ###### Others settings #######
    parser.add_argument('--dataset', type=str, default='citeseer')
    parser.add_argument('--embedding-dim', nargs='+', type=int, default=[3703,16,6])
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    return parser


if __name__ == "__main__":

    parser = parser_loader()
    args = vars(parser.parse_args())
    print(args)

    seed_time = 1
    acc_val = np.zeros(seed_time)
    acc_test = np.zeros(seed_time)
    epoch_list = np.zeros(seed_time)
    for seed in range(seed_time):

        acc_test, epoch = run_pruning_acc(args, seed)
        print("Seed:[{}], Test:[{:.2f}] at epoch:[{}]".format(seed, acc_test[seed] * 100, epoch_list[seed]))

    print('Finish !')
    print("syd:" + "-" * 100)
    print("syd: Pruning Percent:[{}]".format(args['pruning_percent']))
    print('syd: Test mean : [{:.2f}]  std : [{:.2f}]'.format(acc_test.mean() * 100, acc_test.std() * 100))
    print('syd: Mean Epoch: [{:.2f}]'.format(epoch_list.mean()))
    print("syd:" + "-" * 100)