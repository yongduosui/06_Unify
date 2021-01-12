import __init__
from ogb.nodeproppred import Evaluator
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, add_self_loops
from args import ArgsInit
from ogb.nodeproppred import PygNodePropPredDataset
from model import DeeperGCN
from utils.ckpt_util import save_ckpt
import logging
import pruning
import time
import pdb
import warnings
warnings.filterwarnings('ignore')

@torch.no_grad()
def test(model, x, edge_index, y_true, split_idx, evaluator):
    model.eval()
    out = model(x, edge_index)

    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


def train(model, x, edge_index, y_true, train_idx, optimizer, args):

    model.train()
    optimizer.zero_grad()
    pred = model(x, edge_index)[train_idx]
    loss = F.nll_loss(pred, y_true.squeeze(1)[train_idx])
    loss.backward()
    
    pruning.subgradient_update_mask(model, args) # l1 norm
    optimizer.step()
    return loss.item()


def main():

    args = ArgsInit().save_exp()

    if args.use_gpu:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device('cpu')

    dataset = PygNodePropPredDataset(name=args.dataset)
    data = dataset[0]
    split_idx = dataset.get_idx_split()

    evaluator = Evaluator(args.dataset)

    x = data.x.to(device)
    y_true = data.y.to(device)
    train_idx = split_idx['train'].to(device)

    edge_index = data.edge_index.to(device)
    edge_index = to_undirected(edge_index, data.num_nodes)

    if args.self_loop:
        edge_index = add_self_loops(edge_index, num_nodes=data.num_nodes)[0]

    sub_dir = 'SL_{}'.format(args.self_loop)

    args.in_channels = data.x.size(-1)
    args.num_tasks = dataset.num_classes

    model = DeeperGCN(args).to(device)
    pruning.add_mask(model)

    for name, param in model.named_parameters():
        if 'mask' in name:
            if args.fixed == 'all_fixed':
                param.requires_grad = False
            elif args.fixed == 'only_adj':
                if 'edge' in name:
                    param.requires_grad = False
            elif args.fixed == 'only_wei':
                if 'edge' not in name:
                    param.requires_grad = False
            else:
                assert args.fixed == 'no_fixed'

            print("NAME:[{}]\tSHAPE:[{}]\tGRAD:[{}]".format(name, param.shape, param.requires_grad))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    results = {'highest_valid': 0, 'final_train': 0, 'final_test': 0, 'highest_train': 0}
    start_time = time.time()
    test_accuracy = 0.0

    for epoch in range(1, args.epochs + 1):
        
        pruning.plot_mask_distribution(model, epoch, test_accuracy, args.model_save_path + '/plot')

        epoch_loss = train(model, x, edge_index, y_true, train_idx, optimizer, args)
        logging.info('Epoch {}, training loss {:.4f}'.format(epoch, epoch_loss))
        model.print_params(epoch=epoch)

        result = test(model, x, edge_index, y_true, split_idx, evaluator)
        logging.info(result)
        train_accuracy, valid_accuracy, test_accuracy = result

        if train_accuracy > results['highest_train']:
            results['highest_train'] = train_accuracy

        if valid_accuracy > results['highest_valid']:
            results['highest_valid'] = valid_accuracy
            results['final_train'] = train_accuracy
            results['final_test'] = test_accuracy

            save_ckpt(model, optimizer, round(epoch_loss, 4), epoch, args.model_save_path, sub_dir, name_post='valid_best')

        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' +
              'Epoch:[{}/{}]\t Results LOSS:[{:.4f}] Train :[{:.2f}] Valid:[{:.2f}] Test:[{:.2f}] | Update Test:[{:.2f}]'
              .format(epoch, args.epochs, epoch_loss, train_accuracy * 100, valid_accuracy * 100, test_accuracy * 100, results['final_test'] * 100))

    save_ckpt(model, optimizer, round(epoch_loss, 4), epoch, args.model_save_path, sub_dir, name_post='last_epoch')

    end_time = time.time()
    total_time = end_time - start_time
    logging.info('Total time: {}'.format(time.strftime('%H:%M:%S', time.gmtime(total_time))))
    print('-' * 100)
    print("syd : Final Result Train:[{:.2f}]  Valid:[{:.2f}]  Test:[{:.2f}]"
        .format(results['final_train'] * 100, results['highest_valid'] * 100, results['final_test'] * 100))
    print('-' * 100)

if __name__ == "__main__":
    main()
