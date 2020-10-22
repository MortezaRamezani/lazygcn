import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_sparse import spmm
import math


class SAGEConv(nn.Module):

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

    def forward(self, adj, h):

        out_nodes = adj.sizes()[0]

        support = adj.spmm(h)

        self_h = self.linear(h[:out_nodes])
        neigh_h = self.neigh_linear(support)

        if self.concat:
            h = torch.cat([self_h, neigh_h], dim=1)

        if self.activation:
            h = self.activation(h)

        h = self.layer_norm(h)

        return h
