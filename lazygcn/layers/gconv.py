import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_sparse import spmm
import math


class GConv(nn.Module):

    def __init__(self,
                input_dim,
                output_dim,
                layer_id,
                activation=None
                ):
        super().__init__()
        
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = activation

        # nn.init.xavier_uniform_(self.linear.weight)
        # bound = 1.0 / math.sqrt(self.linear.weight.size(0))
        # self.linear.weight.data.uniform_(-bound, bound)
        # self.linear.bias.data.uniform_(-bound, bound)

    def forward(self, adj, h):

        # h = self.linear(h)

        h = adj.spmm(h)
        # h = spmm(adj[0], adj[1], adj[2][0], adj[2][1], h)
        # h = torch.sparse.mm(adj, h)

        h = self.linear(h)
        
        if self.activation:
            h = self.activation(h)
        return h
