import torch

def torch_normalize_adj(adj):
    adj = adj + torch.eye(adj.shape[0]).cuda()
    rowsum = adj.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt).cuda()
    return adj.mm(d_mat_inv_sqrt).t().mm(d_mat_inv_sqrt)