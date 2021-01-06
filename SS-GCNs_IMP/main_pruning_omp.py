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

def run_fix_mask(args, seed, rewind_weight_mask):

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
    net_gcn.load_state_dict(rewind_weight_mask)
    adj_spar, wei_spar = pruning.print_sparsity(net_gcn)
    
    for name, param in net_gcn.named_parameters():
        if 'mask' in name:
            param.requires_grad = False

    optimizer = torch.optim.Adam(net_gcn.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    acc_test = 0.0
    best_val_acc = {'val_acc': 0, 'epoch' : 0, 'test_acc': 0}

    for epoch in range(400):

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
                best_val_acc['val_acc'] = acc_val
                best_val_acc['test_acc'] = acc_test
                best_val_acc['epoch'] = epoch
 
        print("(Fix Mask) Epoch:[{}] Val:[{:.2f}] Test:[{:.2f}] | Final Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}]"
                 .format(epoch, acc_val * 100, acc_test * 100, 
                                best_val_acc['val_acc'] * 100, best_val_acc['test_acc'] * 100, best_val_acc['epoch']))

    return best_val_acc['val_acc'], best_val_acc['test_acc'], best_val_acc['epoch'], adj_spar, wei_spar


def run_get_mask(args, seed, rewind_weight_mask=None):

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
    
    pruning.add_trainable_mask_noise(net_gcn)

    net_gcn = net_gcn.cuda()
    
    if rewind_weight_mask:
        net_gcn.load_state_dict(rewind_weight_mask)
        adj_spar, wei_spar = pruning.print_sparsity(net_gcn)

    optimizer = torch.optim.Adam(net_gcn.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    acc_test = 0.0
    best_val_acc = {'val_acc': 0, 'epoch' : 0, 'test_acc':0}
    
    rewind_weight = copy.deepcopy(net_gcn.state_dict())
    for epoch in range(args['total_epoch']):

        optimizer.zero_grad()
        output = net_gcn(features, adj)
        loss = loss_func(output[idx_train], labels[idx_train])
        loss.backward()
        pruning.subgradient_update_mask(net_gcn, args) # l1 norm
        optimizer.step()
        with torch.no_grad():
            output = net_gcn(features, adj, val_test=True)
            acc_val = f1_score(labels[idx_val].cpu().numpy(), output[idx_val].cpu().numpy().argmax(axis=1), average='micro')
            acc_test = f1_score(labels[idx_test].cpu().numpy(), output[idx_test].cpu().numpy().argmax(axis=1), average='micro')
            if acc_val > best_val_acc['val_acc']:
                best_val_acc['test_acc'] = acc_test
                best_val_acc['val_acc'] = acc_val
                best_val_acc['epoch'] = epoch
                #best_epoch_mask = pruning.get_final_mask_epoch(net_gcn, percent=args['pruning_percent'])

            print("(Get Mask) Epoch:[{}] Val:[{:.2f}] Test:[{:.2f}] | Best Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}]"
                 .format(epoch, acc_val * 100, acc_test * 100, 
                                best_val_acc['val_acc'] * 100,  
                                best_val_acc['test_acc'] * 100,
                                best_val_acc['epoch']))
        
    
    best_epoch_mask = pruning.get_final_mask_epoch(net_gcn, percent=args['pruning_percent'])

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
    seed_dict = {'cora': 307, 'citeseer': 118}
    # seed_time = 30
    # rand_seed_list = np.random.randint(100, 500, seed_time)
    # for seed in rand_seed_list:
    seed = seed_dict[args['dataset']]
    rewind_weight = None
    for p in range(1):
        
        final_mask_dict, rewind_weight = run_get_mask(args, seed, rewind_weight)
        
        rewind_weight['adj_mask1_train'] = final_mask_dict['adj_mask']
        rewind_weight['adj_mask2_fixed'] = final_mask_dict['adj_mask']
        rewind_weight['net_layer.0.weight_mask_train'] = final_mask_dict['weight1_mask']
        rewind_weight['net_layer.0.weight_mask_fixed'] = final_mask_dict['weight1_mask']
        rewind_weight['net_layer.1.weight_mask_train'] = final_mask_dict['weight2_mask']
        rewind_weight['net_layer.1.weight_mask_fixed'] = final_mask_dict['weight2_mask']

        best_acc_val, final_acc_test, final_epoch_list, adj_spar, wei_spar = run_fix_mask(args, seed, rewind_weight)
        print("=" * 120)
        print("syd : Sparsity:[0.9^{}={:.2f}%], Best Val:[{:.2f}] at epoch:[{}] | Final Test Acc:[{:.2f}] Adj:[{:.2f}%] Wei:[{:.2f}%]"
            .format(p + 1, 0.9 ** (p + 1) * 100, best_acc_val * 100, final_epoch_list, final_acc_test * 100, adj_spar, wei_spar))
        print("=" * 120)
