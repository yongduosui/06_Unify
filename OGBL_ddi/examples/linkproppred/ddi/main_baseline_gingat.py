import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling
import torch_geometric.transforms as T
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from logger import Logger
import pdb
import torch
from models import GCN, GCN_yuning, LinkPredictor
from gnns_clean.gin_net import GINNet
from gnns_clean.gat_net import GATNet
import train
import numpy as np
from scipy.sparse import coo_matrix
import dgl
import train_gingat

def main(args):

    device = f'cuda:{args.device}'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name='ogbl-ddi', transform=T.ToSparseTensor())
    data = dataset[0]
    adj_t = data.adj_t.to(device)
    
    node_num = adj_t.size(0)
    adj_row = adj_t.coo()[0]
    adj_col = adj_t.coo()[1]

    g = dgl.DGLGraph()
    g.add_nodes(node_num)
    g.add_edges(adj_row, adj_col)

    split_edge = dataset.get_edge_split()

    # We randomly pick some training samples that we want to evaluate on:
    torch.manual_seed(12345)
    idx = torch.randperm(split_edge['train']['edge'].size(0))
    idx = idx[:split_edge['valid']['edge'].size(0)]
    split_edge['eval_train'] = {'edge': split_edge['train']['edge'][idx]}

    if args.net == 'gin':
        model = GINNet([256, 256, 256]).to(device)
    elif args.net == 'gat':
        model = GATNet([256, 256, 256]).to(device)
        g.add_edges(list(range(node_num)), list(range(node_num)))
        
    else: 
        assert False
        
    emb = torch.nn.Embedding(data.num_nodes, args.hidden_channels).to(device)
    predictor = LinkPredictor(args.hidden_channels, 
                              args.hidden_channels, 1,
                              args.num_layers, 
                              args.dropout).to(device)

    evaluator = Evaluator(name='ogbl-ddi')
    torch.nn.init.xavier_uniform_(emb.weight)
    predictor.reset_parameters()

    optimizer = torch.optim.Adam(
        list(model.parameters()) + 
        list(emb.parameters()) +
        list(predictor.parameters()), lr=args.lr)

    key = 'Hits@20'
    best_val_acc = {'val_acc': 0, 'epoch' : 0, 'test_acc':0, 'best_test': 0}
    for epoch in range(1, args.epochs + 1):
        
        loss = train_gingat.train_yuning(model, predictor, 
                                                emb.weight, 
                                                adj_t, 
                                                g, split_edge, optimizer, args)
        results = train_gingat.test(model, predictor, 
                                           emb.weight, 
                                           adj_t, 
                                           g, split_edge, evaluator, args.batch_size, yuning=True)

        train_hits, valid_hits, test_hits = results[key]

        if valid_hits > best_val_acc['val_acc']:
            best_val_acc['test_acc'] = test_hits
            best_val_acc['val_acc'] = valid_hits
            best_val_acc['epoch'] = epoch

        if test_hits > best_val_acc['best_test']:
            best_val_acc['best_test'] = test_hits

        print("[{}] (Baseline) Epoch:[{}/{}] Loss:[{:.2f}] Train:[{:.2f}] Val:[{:.2f}] Test:[{:.2f}] | Best Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}] | Best Test:[{:.2f}]"
              .format(args.net, epoch, 
                                args.epochs,
                                loss,
                                train_hits * 100, 
                                valid_hits * 100, 
                                test_hits * 100,
                                best_val_acc['val_acc'] * 100,  
                                best_val_acc['test_acc'] * 100,
                                best_val_acc['epoch'],
                                best_val_acc['best_test'] * 100))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='OGBL-DDI (GNN)')

    parser.add_argument('--net', type=str, default='')

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)
    main(args)
