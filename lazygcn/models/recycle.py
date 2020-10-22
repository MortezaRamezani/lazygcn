import torch
import torch.nn as nn
import torch.nn.functional as F

from .gcn import GCN
from ..layers import RecGConv, SAGEConv, RecSAGEConv


class RecycleGCN(GCN):

    def __init__(self,
                 num_layers,
                 features_dim,
                 hidden_dim,
                 num_classes,
                 activation,
                 dropout=0,
                 weight_decay=0,
                 layer=RecGConv,
                 **kwargs):

        super().__init__(num_layers,
                         features_dim,
                         hidden_dim,
                         num_classes,
                         activation,
                         dropout,
                         weight_decay,
                         layer,
                         **kwargs)

    def forward(self, nodeblock, x, recycle_vector=None, full=False):

        h = x
        for i, layer in enumerate(self.layers):
            
            if full:
                nb = nodeblock[0]
            else:
                nb = nodeblock[i]

            if recycle_vector is not None and i == (self.num_layers - 1):
                h = layer(nb, h, recycle_vector)
            else:
                h = layer(nb, h)

        return h

class RecycleSAGENet(nn.Module):

    def __init__(self,
                 num_layers,
                 features_dim,
                 hidden_dim,
                 num_classes,
                 activation,
                 dropout=0,
                 weight_decay=0,
                 layer=RecSAGEConv,#SAGEConv,
                 concat=True):

        super().__init__()

        self.concat = concat
        self.dropout = nn.Dropout(p=dropout) if dropout != 0 else 0
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        self.layers.append(layer(features_dim, hidden_dim, 0, activation, concat=concat))
        for i in range(1, num_layers):
            self.layers.append(layer(2*hidden_dim, hidden_dim, i, activation, concat=concat))

        self.layers.append(nn.Linear(2*hidden_dim, num_classes, bias=False))

    def forward(self, nodeblock, x, recycle_vector=None, full=False):

        h = x
        for i, layer in enumerate(self.layers[:-1]):
            
            if full:
                nb = nodeblock[0]
            else:
                nb = nodeblock[i]

            if recycle_vector is not None and i == (self.num_layers - 1):
                h = layer(nb, h, recycle_vector)
            else:
                h = layer(nb, h)

        # Last linear
        # if recycle_vector is not None:
        #     h = self.layers[-1](h[recycle_vector, :])
        # else:
        #     h = self.layers[-1](h)

        h = self.layers[-1](h)

        return h