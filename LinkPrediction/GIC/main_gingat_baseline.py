"Implementation based on https://github.com/PetarV-/DGI"
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from models import GIC, LogReg, GIC_GCN, GIC_GAT, GIC_GIN
from utils import process
import argparse
import pdb
import pruning
import dgl
import warnings
warnings.filterwarnings('ignore')

def main(args):

    torch.manual_seed(args.seed)
    num_clusters = int(args.num_clusters)
    dataset = args.dataset

    batch_size = 1
    patience = 50
    l2_coef = 0.0
    hid_units = 16
    # sparse = True
    sparse = False
    nonlinearity = 'prelu' # special name to separate parameters

    adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)
    adj_sparse = adj

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
    test_edges, test_edges_false = process.mask_test_edges(adj, test_frac=args.test_rate, val_frac=0.05)
    adj = adj_train

    ylabels = labels
    features, _ = process.preprocess_features(features)

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = labels.shape[1]
    
    g = dgl.DGLGraph()
    g.add_nodes(nb_nodes)
    adj = adj.tocoo()
    g.add_edges(adj.row, adj.col)

    features = torch.FloatTensor(features[np.newaxis]).cuda()
    labels = torch.FloatTensor(labels[np.newaxis]).cuda()

    b_xent = nn.BCEWithLogitsLoss()
    b_bce = nn.BCELoss()

    if args.net == 'gin':
        model = GIC_GIN(nb_nodes, ft_size, hid_units, nonlinearity, num_clusters, 100)
    elif args.net == 'gat':
        model = GIC_GAT(nb_nodes, ft_size, hid_units, nonlinearity, num_clusters, 100)
        g.add_edges(list(range(nb_nodes)), list(range(nb_nodes)))
    else: assert False

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=l2_coef)
    cnt_wait = 0
    best = 1e9
    best_t = 0

    model.cuda()
    best_val_acc = {'val_acc': 0, 'epoch' : 0, 'test_acc':0}
    for epoch in range(args.epochs):
        model.train()
        optimiser.zero_grad()

        idx = np.random.permutation(nb_nodes)
        shuf_fts = features[:, idx, :]
        lbl_1 = torch.ones(batch_size, nb_nodes)
        lbl_2 = torch.zeros(batch_size, nb_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1)
        shuf_fts = shuf_fts.cuda()
        lbl = lbl.cuda()

        logits, logits2  = model(features, shuf_fts, 
                                           g, 
                                           sparse, None, None, None, 100) 
        loss = 0.5 * b_xent(logits, lbl) + 0.5 * b_xent(logits2, lbl) 
        loss.backward()
        optimiser.step()
        with torch.no_grad():
            acc_val, _ = pruning.test(model, features, 
                                             g, 
                                             sparse, 
                                             adj_sparse, 
                                             val_edges, 
                                             val_edges_false)
            acc_test, _ = pruning.test(model, features, 
                                              g, 
                                              sparse, 
                                              adj_sparse, 
                                              test_edges, 
                                              test_edges_false)
            if acc_val > best_val_acc['val_acc']:
                best_val_acc['test_acc'] = acc_test
                best_val_acc['val_acc'] = acc_val
                best_val_acc['epoch'] = epoch
        
        print("Epoch:[{}/{}], Loss:[{:.4f}] Val:[{:.2f}] Test:[{:.2f}] | Best Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}]"
             .format(epoch, args.epochs, 
                            loss, 
                            acc_val * 100, 
                            acc_test * 100, 
                            best_val_acc['val_acc'] * 100,
                            best_val_acc['test_acc'] * 100,
                            best_val_acc['epoch']))

    print('Final (Baseline) Dataset [{}] Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}]'
        .format(args.dataset, 
                best_val_acc['val_acc'] * 100, 
                best_val_acc['test_acc'] * 100,
                best_val_acc['epoch']))

def parser_loader():

    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--epochs', type=int, default=2000, help='')
    parser.add_argument('--lr', type=float, default=0.001, help='')
    parser.add_argument('--net', type=str, default='cora',help='')

    parser.add_argument('--seed', type=int, default=1234, help='')
    parser.add_argument('--d', dest='dataset', type=str, default='cora',help='')
    parser.add_argument('--b', dest='beta', type=int, default=100,help='')
    parser.add_argument('--c', dest='num_clusters', type=float, default=128,help='')
    parser.add_argument('--a', dest='alpha', type=float, default=0.5,help='')
    parser.add_argument('--test_rate', dest='test_rate', type=float, default=0.1,help='')
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parser_loader()
    print(args)
    main(args)
    
