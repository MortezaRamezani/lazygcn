import torch
import torch.nn as nn

from torch_sparse import spmm

from .gconv import GConv



class RecGConv(GConv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, adj, h, recycle_vec=None):

        h = adj.spmm(h)
        # h = spmm(adj[0], adj[1], adj[2][0], adj[2][1], h)

        if recycle_vec != None:
            h = h[recycle_vec, :]

        h = self.linear(h)

        if self.activation:
            h = self.activation(h)
        return h


class RecSAGEConv(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 layer_id,
                 activation=None,
                 layer_norm=True,
                 concat=True):
        super().__init__()

        self.concat = concat
        self.linear = nn.Linear(input_dim, output_dim)
        self.neigh_linear = nn.Linear(input_dim, output_dim)
        self.activation = activation
        # Bias Norm, Meh
        self.layer_norm = nn.LayerNorm(2*output_dim, elementwise_affine=True)

    def forward(self, adj, h, recycle_vec=None):

        # out_nodes = adj.sizes()[0]
        out_nodes = adj.size(0)

        support = adj.spmm(h)

        h = h[:out_nodes]

        if recycle_vec != None:
            support = support[recycle_vec, :]
            h = h[recycle_vec, :]

        self_h = self.linear(h)
        neigh_h = self.neigh_linear(support)

        if self.concat:
            h = torch.cat([self_h, neigh_h], dim=1)

        if self.activation:
            h = self.activation(h)

        h = self.layer_norm(h)

        return h
