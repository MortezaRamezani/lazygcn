
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import Linear

class Model(nn.Module):

    def __init__(self):
        pass


class MLP(nn.Module):

    def __init__(self,
                 num_layers,
                 features_dim,
                 hidden_dim,
                 num_classes,
                 activation,
                 dropout=0,
                 weight_decay=0,
                 layer=Linear):

        super().__init__()

        self.dropout = nn.Dropout(p=dropout) if dropout != 0 else 0
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        self.layers.append(layer(features_dim, hidden_dim, 0, activation))
        for i in range(1, num_layers - 1):
            self.layers.append(layer(hidden_dim, hidden_dim, i, activation))
        self.layers.append(layer(hidden_dim, num_classes, num_layers))

    def forward(self, x):

        h = x
        for _, layer in enumerate(self.layers):
            h = layer(h)

        return h
