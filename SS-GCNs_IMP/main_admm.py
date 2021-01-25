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
import copy 
import warnings
import utils
import pruning
warnings.filterwarnings("ignore")

def run_admm(args, seed, pre_train_model_dict):

    setup_seed(seed)
    adj, features, labels, idx_train, idx_val, idx_test = load_data(args['dataset'])
    adj = adj.to_dense()
    
    node_num = features.size()[0]
    class_num = labels.numpy().max() + 1

    adj = adj.cuda()
    features = features.cuda()
    labels = labels.cuda()
    loss_func = nn.CrossEntropyLoss()
    
    net_gcn = net.net_gcn_admm(embedding_dim=args['embedding_dim'], adj=adj)
    net_gcn.load_state_dict(pre_train_model_dict)
    net_gcn.adj_layer1.requires_grad = True
    net_gcn.adj_layer2.requires_grad = True
    net_gcn = net_gcn.cuda()
    optimizer = torch.optim.Adam(net_gcn.parameters(), lr=0.0001, weight_decay=args['weight_decay'])
    
    for name, param in net_gcn.named_parameters():
        if 'weight' in name:
            param.requires_grad = False
            print("NAME:{}\tGRAD:{}".format(name, param.requires_grad))
    
    adj1 = net_gcn.adj_layer1
    adj2 = net_gcn.adj_layer2
    V1 = copy.deepcopy(adj1)
    V2 = copy.deepcopy(adj2)
    U1 = torch.zeros_like(V1)
    U2 = torch.zeros_like(V2)
    non_zero_idx = adj1.nonzero().shape[0]
    ADMM_times =4
    Total_epochs = 10
    rho = 1e-3
    prune_ratio = 60
    
    for i in range(ADMM_times):
        for epoch in range(Total_epochs):
            optimizer.zero_grad()
            output = net_gcn(features, adj)
            model_loss = loss_func(output[idx_train], labels[idx_train])
            other_loss = torch.norm(adj1 - torch.eye(adj1.shape[0]).cuda() - V1 + U1, p=2) + torch.norm(adj2 - torch.eye(adj2.shape[0]).cuda() - V2 + U2, p=2)
            total_loss = model_loss + rho * other_loss
            total_loss.backward()
            optimizer.step()
        
        V1 = adj1 - torch.eye(adj1.shape[0]).cuda() + U1
        V1 = pruning.prune_adj(V1.detach().cpu(), non_zero_idx, percent=prune_ratio)
        V1 = torch.tensor(V1).cuda()
        V1 = V1 - torch.eye(adj1.shape[0]).cuda()
        U1 = U1 + (adj1 - torch.eye(adj1.shape[0]).cuda()) - V1

        V2 = adj2 - torch.eye(adj2.shape[0]).cuda() + U2
        V2 = pruning.prune_adj(V2.detach().cpu(), non_zero_idx, percent=prune_ratio) 
        V2 = torch.tensor(V2).cuda()
        V2 = V2 - torch.eye(adj2.shape[0]).cuda()
        U2 = U2 + (adj2 - torch.eye(adj2.shape[0]).cuda()) - V2
        print("ADMM:[{}/{}] LOSS:[{:.2f}]".format(i, ADMM_times, total_loss))
    
    
    # adj1 = pruning.get_each_mask_admm(adj1, 0.05)
    # adj2 = pruning.get_each_mask_admm(adj2, 0.05)
    adj1 = pruning.prune_adj(adj1.detach().cpu() - torch.eye(adj1.shape[0]), non_zero_idx, percent=prune_ratio)
    adj2 = pruning.prune_adj(adj2.detach().cpu() - torch.eye(adj2.shape[0]), non_zero_idx, percent=prune_ratio)

    adj1 = torch.tensor(adj1).cuda()
    adj2 = torch.tensor(adj2).cuda()
    return [adj1, adj2]



def run_get_pretrain(args, seed):

    setup_seed(seed)
    adj, features, labels, idx_train, idx_val, idx_test = load_data(args['dataset'])
    adj = adj.to_dense()
    
    node_num = features.size()[0]
    class_num = labels.numpy().max() + 1

    adj = adj.cuda()
    features = features.cuda()
    labels = labels.cuda()
    loss_func = nn.CrossEntropyLoss()
    
    net_gcn = net.net_gcn_admm(embedding_dim=args['embedding_dim'], adj=adj)
    net_gcn.adj_layer1.requires_grad = False
    net_gcn.adj_layer2.requires_grad = False
    
    net_gcn = net_gcn.cuda()
    optimizer = torch.optim.Adam(net_gcn.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    
    acc_test = 0.0
    best_val_acc = {'val_acc': 0, 'epoch' : 0, 'test_acc': 0}
    for epoch in range(100):

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
                pre_train_model_dict = net_gcn.state_dict()
 
        print("(Pre-training) Epoch:[{}] Val:[{:.2f}] Test:[{:.2f}] | Final Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}]"
                 .format(epoch, acc_val * 100, acc_test * 100, 
                                best_val_acc['val_acc'] * 100, 
                                best_val_acc['test_acc'] * 100, 
                                best_val_acc['epoch']))

    return pre_train_model_dict


def run_eval(args, seed, adj_prune):

    setup_seed(seed)
    adj, features, labels, idx_train, idx_val, idx_test = load_data(args['dataset'])
    adj = adj.to_dense()
    
    node_num = features.size()[0]
    class_num = labels.numpy().max() + 1

    adj = adj.cuda()
    features = features.cuda()
    labels = labels.cuda()
    loss_func = nn.CrossEntropyLoss()
    
    net_gcn = net.net_gcn_admm(embedding_dim=args['embedding_dim'], adj=adj)
    
    adj1 = torch.tensor(utils.preprocess_adj(adj_prune[0].cpu().numpy()).todense())
    adj2 = torch.tensor(utils.preprocess_adj(adj_prune[1].cpu().numpy()).todense())
    net_gcn.adj_layer1 = nn.Parameter(adj1.float().cuda(), requires_grad=False)
    net_gcn.adj_layer2 = nn.Parameter(adj2.float().cuda(), requires_grad=False)
   
    net_gcn = net_gcn.cuda()
    optimizer = torch.optim.Adam(net_gcn.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    
    acc_test = 0.0
    best_val_acc = {'val_acc': 0, 'epoch' : 0, 'test_acc': 0}
    for epoch in range(1000):

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
                pre_train_model_dict = net_gcn.state_dict()
 
        print("(ADMM EVAL) Epoch:[{}] Val:[{:.2f}] Test:[{:.2f}] | Final Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}]"
                 .format(epoch, acc_val * 100, acc_test * 100, 
                                best_val_acc['val_acc'] * 100, 
                                best_val_acc['test_acc'] * 100, 
                                best_val_acc['epoch']))



def parser_loader():
    parser = argparse.ArgumentParser(description='Self-Supervised GCN')
    parser.add_argument('--dataset', type=str, default='citeseer')
    parser.add_argument('--embedding-dim', nargs='+', type=int, default=[3703,16,6])
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
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
    seed_dict = {'cora': 2377, 'citeseer': 4428} # DIM: 512, cora: 2829: 81.9, 2377: 81.1    | cite 4417: 72.1,  4428: 71.3
    seed = seed_dict[args['dataset']]
    rewind_weight = None

    pre_train_model_dict = run_get_pretrain(args, seed)
    adj_prune_list = run_admm(args, seed, pre_train_model_dict)
    run_eval(args, seed, adj_prune_list)
    

