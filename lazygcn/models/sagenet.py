import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import SAGEConv
from ..utils.helper import capture_activations, capture_backprops


class SAGENet(nn.Module):

    def __init__(self,
                 num_layers,
                 features_dim,
                 hidden_dim,
                 num_classes,
                 activation,
                 dropout=0,
                 weight_decay=0,
                 layer=SAGEConv,
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

    def forward(self, nodeblock, x, full=False):

        h = x
        for i, layer in enumerate(self.layers[:-1]):
            if not full:
                h = layer(nodeblock[i], h)
            else:
                h = layer(nodeblock[0], h)


        # h = F.normalize(h, p=2, dim=1)

        # Last linear
        h = self.layers[-1](h)

        return h