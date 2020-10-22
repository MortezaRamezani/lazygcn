import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_sparse import spmm
import math

#TODO: https://github.com/MortezaRamezani/ngnn/blob/master/ngnn/models/batch_asgcn.py

class AdapGConv(nn.Module):

    def __init__(self,
                input_dim,
                output_dim,
                layer_id,
                activation=None):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = activation


    def forward(self, adj, hidden_feat, num_sampled_nodes, q_probs, var_loss=False, is_test=False):

        
        if not is_test:
            h = hidden_feat / q_probs.view(-1,1) / num_sampled_nodes
        else:
            h = hidden_feat
            
        # aggregate
        h = adj.spmm(h)

        # combine
        h = self.linear(h)

        if self.activation:
            h = self.activation(h)
        
        if var_loss and not is_test:
            pre_sup = self.linear(hidden_feat)
            # pre_sup = h
            num_nodes = adj.sizes()[1]
            adj_sup_mean = adj.to_dense().sum(0) / num_nodes
            mu_v = torch.mean(h, dim=0)
            diff = torch.reshape(adj_sup_mean, [-1,1]) * pre_sup - torch.reshape(mu_v, [1,-1])
            vl = torch.sum(diff * diff) / len(h) / num_nodes
            return h, vl

        return h