import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import GConv
from ..utils.helper import capture_activations, capture_backprops


class GCN(nn.Module):

    def __init__(self,
                 num_layers,
                 features_dim,
                 hidden_dim,
                 num_classes,
                 activation,
                 dropout=0,
                 weight_decay=0,
                 layer=GConv,
                 **kwargs):

        super().__init__()

        self.dropout = nn.Dropout(p=dropout) if dropout != 0 else 0
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        self.layers.append(layer(features_dim, hidden_dim, 0, activation))
        for i in range(1, num_layers - 1):
            self.layers.append(layer(hidden_dim, hidden_dim, i, activation))
        self.layers.append(layer(hidden_dim, num_classes, num_layers))

    def forward(self, nodeblock, x, full=False):

        h = x
        for i, layer in enumerate(self.layers):
            if not full:
                h = layer(nodeblock[i], h)
            else:
                # nodeblock is single block (aka full graph adj)
                h = layer(nodeblock[0], h)

        return h


class VRGCN(GCN):
    """ GCN with additional linear layer at the end
    Useful when trying to get per-sample gradients
    """

    def __init__(self,
                 num_layers,
                 features_dim,
                 hidden_dim,
                 num_classes,
                 activation,
                 dropout=0,
                 weight_decay=0,
                 layer=GConv,
                 **kwargs):

        super().__init__(num_layers,
                         features_dim,
                         hidden_dim,
                         num_classes,
                         activation,
                         dropout=0,
                         weight_decay=0,
                         layer=GConv)

        self.num_layers = num_layers + 1
        self.layers = nn.ModuleList()

        self.layers.append(layer(features_dim, hidden_dim, 0, activation))
        for i in range(1, num_layers):
            self.layers.append(layer(hidden_dim, hidden_dim, i, activation))
        self.layers.append(nn.Linear(hidden_dim, num_classes))

    def forward(self, nodeblock, x, full=False):

        h = x

        for i, layer in enumerate(self.layers[:-1]):
            if not full:
                h = layer(nodeblock[i], h)
            else:
                # nodeblock is single block (aka full graph adj)
                h = layer(nodeblock[0], h)

        h = self.layers[-1](h)

        return h
