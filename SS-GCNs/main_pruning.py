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

def run_fix_mask(args, seed, rewind_weight):

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
    net_gcn.load_state_dict(rewind_weight)
    
    for name, param in net_gcn.named_parameters():
        if 'mask' in name:
            param.requires_grad = False
            # print("{}\{} require_grad=False".format(name, param.shape))
    optimizer = torch.optim.Adam(net_gcn.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    acc_test = 0.0
    best_acc = {'best_acc': 0, 'best_epoch' : 0, 'early_stop_acc': 0, 'early_stop_epoch': 0}
    loss_val = []
    for epoch in range(300):
        # pruning.plot_mask_distribution(net_gcn, epoch, acc_test, "mask_distribution")
        optimizer.zero_grad()
        output = net_gcn(features, adj)
        loss = loss_func(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            output = net_gcn(features, adj, val_test=True)
            loss_val.append(loss_func(output[idx_val], labels[idx_val]).cpu().numpy())
            acc_test = f1_score(labels[idx_test].cpu().numpy(), output[idx_test].cpu().numpy().argmax(axis=1), average='micro')
            if acc_test > best_acc['best_acc']:
                best_acc['best_acc'] = acc_test
                best_acc['best_epoch'] = epoch

        if epoch > early_stopping and loss_val[-1] > np.mean(loss_val[-(early_stopping+1):-1]):
            best_acc['early_stop_acc'] = f1_score(labels[idx_test].cpu().numpy(), output[idx_test].cpu().numpy().argmax(axis=1), average='micro')
            best_acc['early_stop_epoch'] = epoch
        print("(Fix Mask) Epoch:[{}] Test Acc[{:.2f}] | Best Acc:[{:.2f}] at Epoch:[{}]"
                 .format(epoch, acc_test * 100, best_acc['best_acc'] * 100, best_acc['best_epoch']))

    return best_acc['best_acc'], best_acc['best_epoch'], best_acc['early_stop_acc'], best_acc['early_stop_epoch']


def run_get_mask(args, seed):

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
    best_acc = {'acc': 0, 'epoch' : 0}

    rewind_weight = copy.deepcopy(net_gcn.state_dict())
    for epoch in range(args['total_epoch']):
        # pruning.plot_mask_distribution(net_gcn, epoch, acc_test, "mask_distribution")
        optimizer.zero_grad()
        output = net_gcn(features, adj)
        loss = loss_func(output[idx_train], labels[idx_train])
        loss.backward()
        pruning.subgradient_update_mask(net_gcn, args) # l1 norm
        optimizer.step()
        with torch.no_grad():
            output = net_gcn(features, adj, val_test=True)
            acc_test = f1_score(labels[idx_test].cpu().numpy(), output[idx_test].cpu().numpy().argmax(axis=1), average='micro')
            if acc_test > best_acc['acc']:
                best_acc['acc'] = acc_test
                best_acc['epoch'] = epoch
                best_epoch_mask = pruning.get_final_mask_epoch(net_gcn, percent=args['pruning_percent'])
            print("(Get Mask) Epoch:[{}] Test Acc[{:.2f}] | Best Acc:[{:.2f}] at Epoch:[{}]"
                 .format(epoch, acc_test * 100, best_acc['acc'] * 100, best_acc['epoch']))
    # final_mask_dict = pruning.get_final_mask(net_gcn, percent=args['pruning_percent'])
    return best_epoch_mask, rewind_weight


def parser_loader():
    parser = argparse.ArgumentParser(description='Self-Supervised GCN')
    ###### Unify pruning settings #######
    parser.add_argument('--s1', type=float, default=0.0001,help='scale sparse rate (default: 0.0001)')
    parser.add_argument('--s2', type=float, default=0.0001,help='scale sparse rate (default: 0.0001)')
    parser.add_argument('--total_epoch', type=int, default=300)
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

    seed_time = 20
    best_acc_test = np.zeros(seed_time)
    early_acc_test = np.zeros(seed_time)
    best_epoch_list = np.zeros(seed_time)
    early_epoch_list = np.zeros(seed_time)

    for seed in range(seed_time):

        final_mask_dict, rewind_weight = run_get_mask(args, seed)
        rewind_weight['adj_mask'] = final_mask_dict['adj_mask']
        rewind_weight['net_layer.0.weight_mask_weight'] = final_mask_dict['weight1_mask']
        rewind_weight['net_layer.1.weight_mask_weight'] = final_mask_dict['weight2_mask']

        best_acc_test[seed], best_epoch_list[seed], early_acc_test[seed], early_epoch_list[seed] = run_fix_mask(args, seed, rewind_weight)
        print("Seed:[{}], BestAcc:[{:.2f}] at epoch:[{}] | EarlyAcc:[{:.2f}] at epoch:[{}]"
            .format(seed, best_acc_test[seed] * 100, best_epoch_list[seed], early_acc_test[seed] * 100, early_epoch_list[seed]))

    print('Finish !')
    print("syd:" + "-" * 100)
    print("syd: Pruning Percent : [{}]".format(args['pruning_percent']))
    print('syd: Best  Acc  mean : [{:.2f}]  std : [{:.2f}] | Epoch: [{}]'.format(best_acc_test.mean() * 100, best_acc_test.std() * 100, best_epoch_list.mean()))
    print('syd: Early Acc  mean : [{:.2f}]  std : [{:.2f}] | Epoch: [{}]'.format(early_acc_test.mean() * 100, early_acc_test.std() * 100, early_epoch_list.mean()))
    print("syd:" + "-" * 100)











    # seed_time = 20
    # acc_val = np.zeros(seed_time)
    # acc_test = np.zeros(seed_time)
    # epoch_list = np.zeros(seed_time)
    # for seed in range(seed_time):

    #     final_mask_dict, rewind_weight = run_get_mask(args, seed)

    #     rewind_weight['adj_mask'] = final_mask_dict['adj_mask']
    #     rewind_weight['net_layer.0.weight_mask_weight'] = final_mask_dict['weight1_mask']
    #     rewind_weight['net_layer.1.weight_mask_weight'] = final_mask_dict['weight2_mask']

    #     acc_val[seed], acc_test[seed], epoch_list[seed] = run_fix_mask(args, seed, rewind_weight)
    #     print("Seed:[{}], Val:[{:.2f}], Test:[{:.2f}] at epoch:[{}]".format(seed, acc_val[seed] * 100, acc_test[seed] * 100, epoch_list[seed]))

    # print('Finish !')
    # print("syd:" + "-" * 100)
    # print("syd: Pruning Percent:[{}]".format(args['pruning_percent']))
    # print('syd: Val  mean : [{:.2f}]  std : [{:.2f}]'.format(acc_val.mean() * 100, acc_val.std() * 100))
    # print('syd: Test mean : [{:.2f}]  std : [{:.2f}]'.format(acc_test.mean() * 100, acc_test.std() * 100))
    # print('syd: Mean Epoch: [{:.2f}]'.format(epoch_list.mean()))
    # print("syd:" + "-" * 100)