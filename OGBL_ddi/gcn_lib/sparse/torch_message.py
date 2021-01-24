import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter, scatter_softmax
from torch_geometric.utils import degree
from torch.nn import Parameter
from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch import Tensor
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul
from torch_geometric.nn import GCNConv, SAGEConv
import math
import pdb

class MyGCNConv(MessagePassing):

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(MyGCNConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.fuse = True
        self._cached_edge_index = None
        self._cached_adj_t = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.edge_mask1_train = None
        

    def reset_parameters(self):

        glorot(self.weight)
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x, edge_index, edge_mask1_train=None, edge_weight=None):
        
        self.edge_mask1_train = edge_mask1_train
        x = torch.matmul(x, self.weight)
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        if self.bias is not None:
            out += self.bias
        return out


    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        
        if not self.edge_mask1_train:
            adj_t = mul(adj_t, self.edge_mask1_train)
        return matmul(adj_t, x, reduce=self.aggr)

              
    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)



def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


class GenMessagePassing(MessagePassing):
    def __init__(self, aggr='softmax',
                 t=1.0, learn_t=False,
                 p=1.0, learn_p=False,
                 y=0.0, learn_y=False):

        if aggr in ['softmax_sg', 'softmax', 'softmax_sum']:

            super(GenMessagePassing, self).__init__(aggr=None)
            self.aggr = aggr

            if learn_t and (aggr == 'softmax' or aggr == 'softmax_sum'):
                self.learn_t = True
                self.t = torch.nn.Parameter(torch.Tensor([t]), requires_grad=True)
            else:
                self.learn_t = False
                self.t = t

            if aggr == 'softmax_sum':
                self.y = torch.nn.Parameter(torch.Tensor([y]), requires_grad=learn_y)

        elif aggr in ['power', 'power_sum']:

            super(GenMessagePassing, self).__init__(aggr=None)
            self.aggr = aggr

            if learn_p:
                self.p = torch.nn.Parameter(torch.Tensor([p]), requires_grad=True)
            else:
                self.p = p

            if aggr == 'power_sum':
                self.y = torch.nn.Parameter(torch.Tensor([y]), requires_grad=learn_y)
        else:
            super(GenMessagePassing, self).__init__(aggr=aggr)

    def aggregate(self, inputs, index, ptr=None, dim_size=None):

        if self.aggr in ['add', 'mean', 'max', None]:
            return super(GenMessagePassing, self).aggregate(inputs, index, ptr, dim_size)

        elif self.aggr in ['softmax_sg', 'softmax', 'softmax_sum']:

            if self.learn_t:
                out = scatter_softmax(inputs*self.t, index, dim=self.node_dim)
            else:
                with torch.no_grad():
                    out = scatter_softmax(inputs*self.t, index, dim=self.node_dim)

            out = scatter(inputs*out, index, dim=self.node_dim,
                          dim_size=dim_size, reduce='sum')

            if self.aggr == 'softmax_sum':
                self.sigmoid_y = torch.sigmoid(self.y)
                degrees = degree(index, num_nodes=dim_size).unsqueeze(1)
                out = torch.pow(degrees, self.sigmoid_y) * out

            return out


        elif self.aggr in ['power', 'power_sum']:
            min_value, max_value = 1e-7, 1e1
            torch.clamp_(inputs, min_value, max_value)
            out = scatter(torch.pow(inputs, self.p), index, dim=self.node_dim,
                          dim_size=dim_size, reduce='mean')
            torch.clamp_(out, min_value, max_value)
            out = torch.pow(out, 1/self.p)

            if self.aggr == 'power_sum':
                self.sigmoid_y = torch.sigmoid(self.y)
                degrees = degree(index, num_nodes=dim_size).unsqueeze(1)
                out = torch.pow(degrees, self.sigmoid_y) * out

            return out

        else:
            raise NotImplementedError('To be implemented')


class MsgNorm(torch.nn.Module):
    def __init__(self, learn_msg_scale=False):
        super(MsgNorm, self).__init__()

        self.msg_scale = torch.nn.Parameter(torch.Tensor([1.0]),
                                            requires_grad=learn_msg_scale)

    def forward(self, x, msg, p=2):
        msg = F.normalize(msg, p=p, dim=1)
        x_norm = x.norm(p=p, dim=1, keepdim=True)
        msg = msg * x_norm * self.msg_scale
        return msg
