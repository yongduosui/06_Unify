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
import train

def main():

    parser = argparse.ArgumentParser(description='OGBL-DDI (GNN)')
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

    device = f'cuda:{args.device}'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name='ogbl-ddi', transform=T.ToSparseTensor())
    data = dataset[0]

    adj_t = data.adj_t.to(device)
    split_edge = dataset.get_edge_split()

    # We randomly pick some training samples that we want to evaluate on:
    torch.manual_seed(12345)
    idx = torch.randperm(split_edge['train']['edge'].size(0))
    idx = idx[:split_edge['valid']['edge'].size(0)]
    split_edge['eval_train'] = {'edge': split_edge['train']['edge'][idx]}

    # model = GCN(args.hidden_channels, args.hidden_channels,
    #             args.hidden_channels, args.num_layers,
    #             args.dropout).to(device)
    model = GCN_yuning(embedding_dim=[256, 256, 256]).to(device)
    emb = torch.nn.Embedding(data.num_nodes, args.hidden_channels).to(device)
    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='ogbl-ddi')
    torch.nn.init.xavier_uniform_(emb.weight)
    predictor.reset_parameters()

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(emb.parameters()) +
        list(predictor.parameters()), lr=args.lr)

    key = 'Hits@20'
    best_val_acc = {'val_acc': 0, 'epoch' : 0, 'test_acc':0}
    for epoch in range(1, 1 + args.epochs):
        
        loss = train.train_yuning(model, predictor, emb.weight, adj_t, split_edge, optimizer, args.batch_size)
        results = train.test(model, predictor, emb.weight, adj_t, split_edge, evaluator, args.batch_size, yuning=True)
        train_hits, valid_hits, test_hits = results[key]

        if valid_hits > best_val_acc['val_acc']:
            best_val_acc['test_acc'] = test_hits
            best_val_acc['val_acc'] = valid_hits
            best_val_acc['epoch'] = epoch

        print("(Baseline) Epoch:[{}/{}] Loss:[{:.2f}] Train:[{:.2f}] Val:[{:.2f}] Test:[{:.2f}] | Best Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}]"
                 .format(epoch, args.epochs,
                                loss,
                                train_hits * 100, 
                                valid_hits * 100, 
                                test_hits * 100,
                                best_val_acc['val_acc'] * 100,  
                                best_val_acc['test_acc'] * 100,
                                best_val_acc['epoch']))


if __name__ == "__main__":
    main()
