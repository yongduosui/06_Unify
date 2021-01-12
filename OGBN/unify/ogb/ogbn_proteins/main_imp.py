import __init__
import torch
import torch.optim as optim
import statistics
from dataset import OGBNDataset
from model import DeeperGCN
from args import ArgsInit
import time
import numpy as np
from ogb.nodeproppred import Evaluator
from utils.ckpt_util import save_ckpt
from utils.data_util import intersection, process_indexes
import logging
import pruning
import copy
import pdb
import warnings
warnings.filterwarnings('ignore')

def train(data, dataset, model, optimizer, criterion, device, args):

    loss_list = []
    model.train()
    sg_nodes, sg_edges, sg_edges_index, _ = data
    train_y = dataset.y[dataset.train_idx]
    idx_clusters = np.arange(len(sg_nodes))
    np.random.shuffle(idx_clusters)

    for idx in idx_clusters:

        x = dataset.x[sg_nodes[idx]].float().to(device)
        sg_nodes_idx = torch.LongTensor(sg_nodes[idx]).to(device)
        sg_edges_ = sg_edges[idx].to(device)
        sg_edges_attr = dataset.edge_attr[sg_edges_index[idx]].to(device)
        mapper = {node: idx for idx, node in enumerate(sg_nodes[idx])}
        inter_idx = intersection(sg_nodes[idx], dataset.train_idx.tolist())
        training_idx = [mapper[t_idx] for t_idx in inter_idx]
        optimizer.zero_grad()
        pred = model(x, sg_nodes_idx, sg_edges_, sg_edges_attr)
        target = train_y[inter_idx].to(device)
        loss = criterion(pred[training_idx].to(torch.float32), target.to(torch.float32))
        loss.backward()
        pruning.subgradient_update_mask(model, args) # l1 norm
        optimizer.step()
        loss_list.append(loss.item())

    return statistics.mean(loss_list)

def train_fixed(data, dataset, model, optimizer, criterion, device, args):

    loss_list = []
    model.train()
    sg_nodes, sg_edges, sg_edges_index, _ = data
    train_y = dataset.y[dataset.train_idx]
    idx_clusters = np.arange(len(sg_nodes))
    np.random.shuffle(idx_clusters)

    for idx in idx_clusters:

        x = dataset.x[sg_nodes[idx]].float().to(device)
        sg_nodes_idx = torch.LongTensor(sg_nodes[idx]).to(device)
        sg_edges_ = sg_edges[idx].to(device)
        sg_edges_attr = dataset.edge_attr[sg_edges_index[idx]].to(device)
        mapper = {node: idx for idx, node in enumerate(sg_nodes[idx])}
        inter_idx = intersection(sg_nodes[idx], dataset.train_idx.tolist())
        training_idx = [mapper[t_idx] for t_idx in inter_idx]
        optimizer.zero_grad()
        pred = model(x, sg_nodes_idx, sg_edges_, sg_edges_attr)
        target = train_y[inter_idx].to(device)
        loss = criterion(pred[training_idx].to(torch.float32), target.to(torch.float32))
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

    return statistics.mean(loss_list)

@torch.no_grad()
def multi_evaluate(valid_data_list, dataset, model, evaluator, device):
    model.eval()
    target = dataset.y.detach().numpy()

    train_pre_ordered_list = []
    valid_pre_ordered_list = []
    test_pre_ordered_list = []

    test_idx = dataset.test_idx.tolist()
    train_idx = dataset.train_idx.tolist()
    valid_idx = dataset.valid_idx.tolist()

    for valid_data_item in valid_data_list:
        sg_nodes, sg_edges, sg_edges_index, _ = valid_data_item
        idx_clusters = np.arange(len(sg_nodes))

        test_predict = []
        test_target_idx = []

        train_predict = []
        valid_predict = []

        train_target_idx = []
        valid_target_idx = []

        for idx in idx_clusters:
            x = dataset.x[sg_nodes[idx]].float().to(device)
            sg_nodes_idx = torch.LongTensor(sg_nodes[idx]).to(device)

            mapper = {node: idx for idx, node in enumerate(sg_nodes[idx])}
            sg_edges_attr = dataset.edge_attr[sg_edges_index[idx]].to(device)

            inter_tr_idx = intersection(sg_nodes[idx], train_idx)
            inter_v_idx = intersection(sg_nodes[idx], valid_idx)

            train_target_idx += inter_tr_idx
            valid_target_idx += inter_v_idx

            tr_idx = [mapper[tr_idx] for tr_idx in inter_tr_idx]
            v_idx = [mapper[v_idx] for v_idx in inter_v_idx]

            pred = model(x, sg_nodes_idx, sg_edges[idx].to(device), sg_edges_attr).cpu().detach()

            train_predict.append(pred[tr_idx])
            valid_predict.append(pred[v_idx])

            inter_te_idx = intersection(sg_nodes[idx], test_idx)
            test_target_idx += inter_te_idx

            te_idx = [mapper[te_idx] for te_idx in inter_te_idx]
            test_predict.append(pred[te_idx])

        train_pre = torch.cat(train_predict, 0).numpy()
        valid_pre = torch.cat(valid_predict, 0).numpy()
        test_pre = torch.cat(test_predict, 0).numpy()

        train_pre_ordered = train_pre[process_indexes(train_target_idx)]
        valid_pre_ordered = valid_pre[process_indexes(valid_target_idx)]
        test_pre_ordered = test_pre[process_indexes(test_target_idx)]

        train_pre_ordered_list.append(train_pre_ordered)
        valid_pre_ordered_list.append(valid_pre_ordered)
        test_pre_ordered_list.append(test_pre_ordered)

    train_pre_final = torch.mean(torch.Tensor(train_pre_ordered_list), dim=0)
    valid_pre_final = torch.mean(torch.Tensor(valid_pre_ordered_list), dim=0)
    test_pre_final = torch.mean(torch.Tensor(test_pre_ordered_list), dim=0)

    eval_result = {}

    input_dict = {"y_true": target[train_idx], "y_pred": train_pre_final}
    eval_result["train"] = evaluator.eval(input_dict)

    input_dict = {"y_true": target[valid_idx], "y_pred": valid_pre_final}
    eval_result["valid"] = evaluator.eval(input_dict)

    input_dict = {"y_true": target[test_idx], "y_pred": test_pre_final}
    eval_result["test"] = evaluator.eval(input_dict)

    return eval_result


def main_get_mask(args, imp_num, rewind_weight_mask=None, resume_train_ckpt=None):
    
    device = torch.device("cuda:" + str(args.device))
    dataset = OGBNDataset(dataset_name=args.dataset)
    # extract initial node features
    nf_path = dataset.extract_node_features(args.aggr)

    args.num_tasks = dataset.num_tasks
    args.nf_path = nf_path
    evaluator = Evaluator(args.dataset)
    criterion = torch.nn.BCEWithLogitsLoss()

    valid_data_list = []

    for i in range(args.num_evals):
        parts = dataset.random_partition_graph(dataset.total_no_of_nodes,
                                               cluster_number=args.valid_cluster_number)
        valid_data = dataset.generate_sub_graphs(parts,
                                                 cluster_number=args.valid_cluster_number)
        valid_data_list.append(valid_data)

    sub_dir = 'random-train_{}-test_{}-num_evals_{}'.format(args.cluster_number,
                                                            args.valid_cluster_number,
                                                            args.num_evals)
    logging.info(sub_dir)

    model = DeeperGCN(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    results = {'highest_valid': 0,
               'final_train': 0,
               'final_test': 0,
               'highest_train': 0}

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # do random partition every epoch
        train_parts = dataset.random_partition_graph(dataset.total_no_of_nodes,
                                                     cluster_number=args.cluster_number)
        data = dataset.generate_sub_graphs(train_parts, cluster_number=args.cluster_number)

        epoch_loss = train(data, dataset, model, optimizer, criterion, device)
        logging.info('Epoch {}, training loss {:.4f}'.format(epoch, epoch_loss))

        model.print_params(epoch=epoch)

        result = multi_evaluate(valid_data_list, dataset, model, evaluator, device)

        if epoch % 5 == 0:
            logging.info('%s' % result)

        train_result = result['train']['rocauc']
        valid_result = result['valid']['rocauc']
        test_result = result['test']['rocauc']

        if valid_result > results['highest_valid']:
            results['highest_valid'] = valid_result
            results['final_train'] = train_result
            results['final_test'] = test_result

            save_ckpt(model, optimizer, round(epoch_loss, 4),
                      epoch,
                      args.model_save_path, sub_dir,
                      name_post='valid_best')

        if train_result > results['highest_train']:
            results['highest_train'] = train_result

    logging.info("%s" % results)

    end_time = time.time()
    total_time = end_time - start_time
    logging.info('Total time: {}'.format(time.strftime('%H:%M:%S', time.gmtime(total_time))))


if __name__ == "__main__":

    args = ArgsInit().save_exp()
    pruning.print_args(args, 120)

    start_imp = 1
    rewind_weight_mask = None
    resume_train_ckpt = None

    if args.resume_dir:
        resume_train_ckpt = torch.load(args.resume_dir)
        start_imp = resume_train_ckpt['imp_num']
        rewind_weight_mask = resume_train_ckpt['rewind_weight_mask']

        if 'fixed_ckpt' in args.resume_dir:
            main_fixed_mask(args, start_imp, rewind_weight_mask, resume_train_ckpt)
            start_imp += 1

    for imp_num in range(start_imp, 21):

        rewind_weight_mask = main_get_mask(args, imp_num, rewind_weight_mask, resume_train_ckpt)
        main_fixed_mask(args, imp_num, rewind_weight_mask)