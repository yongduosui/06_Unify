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
from models import GCN, GCN_yuning, GCN_IMP, LinkPredictor
import train
import copy
import pruning

def run_fix_mask(args, imp_num, rewind_weight_mask):

    pruning.setup_seed(args.seed)
    device = f'cuda:{args.device}'
    device = torch.device(device)
    dataset = PygLinkPropPredDataset(name='ogbl-ddi', transform=T.ToSparseTensor())
    data = dataset[0]
    adj_t = data.adj_t.to(device)
    split_edge = dataset.get_edge_split()

    idx = torch.randperm(split_edge['train']['edge'].size(0))
    idx = idx[:split_edge['valid']['edge'].size(0)]
    split_edge['eval_train'] = {'edge': split_edge['train']['edge'][idx]}

    model = GCN_IMP(embedding_dim=[256, 256, 256], adj=adj_t).to(device)
    pruning.add_mask(model)

    emb = torch.nn.Embedding(data.num_nodes, args.hidden_channels).to(device)
    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='ogbl-ddi')
    torch.nn.init.xavier_uniform_(emb.weight)
    predictor.reset_parameters()

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(emb.parameters()) +
        list(predictor.parameters()), lr=args.lr)

    # rewind all weight and masks
    model.load_state_dict(rewind_weight_mask['model'])
    predictor.load_state_dict(rewind_weight_mask['predictor'])
    adj_spar, wei_spar = pruning.print_sparsity(model)

    for name, param in model.named_parameters():
        if 'mask' in name:
            param.requires_grad = False

    key = 'Hits@20'
    best_val_acc = {'val_acc': 0, 'epoch' : 0, 'test_acc':0}
    for epoch in range(1, args.fix_epoch + 1):
        
        loss = train.train_fixed(model, predictor, emb.weight, adj_t, split_edge, optimizer, args)
        results = train.test(model, predictor, emb.weight, adj_t, split_edge, evaluator, args.batch_size, yuning=True)
        train_hits, valid_hits, test_hits = results[key]

        if valid_hits > best_val_acc['val_acc']:
            best_val_acc['test_acc'] = test_hits
            best_val_acc['val_acc'] = valid_hits
            best_val_acc['epoch'] = epoch

        print("IMP:[{}] (Fix Mask) Epoch:[{}/{}] Loss:[{:.2f}] Train:[{:.2f}] Val:[{:.2f}] Test:[{:.2f}] | Best Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}]"
               .format(imp_num, epoch, args.fix_epoch,
                                loss,
                                train_hits * 100, 
                                valid_hits * 100, 
                                test_hits * 100,
                                best_val_acc['val_acc'] * 100,  
                                best_val_acc['test_acc'] * 100,
                                best_val_acc['epoch']))

    print("=" * 120)
    print("syd final: IMP:[{}] Best Val:[{:.2f}] at epoch:[{}] | Final Test Acc:[{:.2f}] Adj:[{:.2f}%] Wei:[{:.2f}%]"
        .format(imp_num, 
                best_val_acc['val_acc'] * 100, 
                best_val_acc['epoch'], 
                best_val_acc['test_acc'] * 100, 
                adj_spar * 100, 
                wei_spar * 100))
    print("=" * 120)


if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(description='OGBL-DDI (GNN)')
    ###### Unify pruning settings #######
    parser.add_argument('--s1', type=float, default=0.0001,help='scale sparse rate (default: 0.0001)')
    parser.add_argument('--s2', type=float, default=0.0001,help='scale sparse rate (default: 0.0001)')
    parser.add_argument('--mask_epoch', type=int, default=200)
    parser.add_argument('--fix_epoch', type=int, default=200)
    parser.add_argument('--pruning_percent_wei', type=float, default=0.2)
    parser.add_argument('--pruning_percent_adj', type=float, default=0.05)
    parser.add_argument('--weight_dir', type=str, default='')
    parser.add_argument('--rewind_soft_mask', action='store_true')
    parser.add_argument('--seed', type=int, default=666)

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
    pruning.print_args(args)

    rewind_weight = None
    for imp_num in range(1, 21):

        final_mask_dict, rewind_weight = run_get_mask(args, imp_num, rewind_weight)
        
        rewind_weight['model']['adj_mask1_train'] = final_mask_dict['adj_mask']
        rewind_weight['model']['adj_mask2_fixed'] = final_mask_dict['adj_mask']
        rewind_weight['model']['net_layer.0.weight_mask_train'] = final_mask_dict['weight1_mask']
        rewind_weight['model']['net_layer.0.weight_mask_fixed'] = final_mask_dict['weight1_mask']
        rewind_weight['model']['net_layer.1.weight_mask_train'] = final_mask_dict['weight2_mask']
        rewind_weight['model']['net_layer.1.weight_mask_fixed'] = final_mask_dict['weight2_mask']

        run_fix_mask(args, imp_num, rewind_weight)

        
