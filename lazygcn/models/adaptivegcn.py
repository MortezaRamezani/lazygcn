import torch
import torch.nn as nn
import torch.nn.functional as F

from .gcn import GCN
from ..layers import AdapGConv

# TODO: https://github.com/MortezaRamezani/ngnn/blob/master/ngnn/models/batch_asgcn.py


class AdaptiveGCN(GCN):

    def __init__(self,
                 num_layers,
                 features_dim,
                 hidden_dim,
                 num_classes,
                 activation,
                 dropout=0,
                 weight_decay=0,
                 layer=AdapGConv,
                 **kwargs):

        super().__init__(num_layers,
                         features_dim,
                         hidden_dim,
                         num_classes,
                         activation,
                         dropout,
                         weight_decay,
                         layer)
        # No need for sampled weight when no attention

    def forward(self, nodeblock, x, is_test=False, full=False):

        h = x
        var_loss = 0

        for i, layer in enumerate(self.layers):

            if full:
                nb = nodeblock[0]
                sn = [] #nodeblock.sampled_nodes[0]
                qp = None
                # TODO: assuming full is only used for when it's test/val
                # ! don't use adaptive with fullgraph for training yet!
                is_test = True
            else:
                nb = nodeblock[i]
                sn = nodeblock.sampled_nodes[i]
                qp = nodeblock.sampled_nodes_aux[i]


            if self.dropout:
                h = self.dropout(h)

            if not is_test:
                if i == len(self.layers) - 1:
                    h, var_loss = layer(nb, h, len(sn), qp,
                                        var_loss=True, is_test=is_test)
                else:
                    h = layer(nb, h, len(sn), qp,
                              var_loss=False, is_test=is_test)
            else:
                h = layer(nb, h, len(sn), qp, var_loss=False, is_test=is_test)

        if is_test:
            return h

        return h, var_loss
