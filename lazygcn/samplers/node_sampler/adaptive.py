
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from ...graph import NodeBlock
from ...utils.helper import scipy_to_sptensor, row_normalize, adj_to_indval

# TODO: https://github.com/MortezaRamezani/ngnn/blob/master/ngnn/samplers/adaptive.py

class AdaptiveNodeSampler(object):

    def __init__(self, graph, num_layers, sapmles_per_layer, *args, **kwargs):
        self.graph = graph
        self.num_layers = num_layers
        self.sapmles_per_layer = sapmles_per_layer

        self.features = kwargs['features']


    def __call__(self, seed, *args, **kwargs):
        
        if callable(seed):
            seed = seed()
        
        sample_weights = kwargs['sample_weights']

        nodeblock = NodeBlock()
        prev_nodes = seed
        nodeblock.sampled_nodes.insert(0, prev_nodes)
        
        for depth in range(self.num_layers):
            current_layer_adj = self.graph.adj[prev_nodes, :]
            neighbor_nodes = np.unique(current_layer_adj.indices)
            
            sparse_adj = current_layer_adj[:, neighbor_nodes]
            square_adj = sparse_adj.multiply(sparse_adj).sum(0)
            tensor_adj = torch.FloatTensor(square_adj.A[0])
        
            x_u = self.features[neighbor_nodes]
            h_u = torch.mm(x_u, sample_weights)
            p_u = F.relu(h_u.flatten()) + 1

            adj_part = torch.sqrt(tensor_adj) #.cuda()
            probas = adj_part * p_u
            probas = probas / torch.sum(probas)
            
            # debug
            probas = probas.cpu()

            sample_size = min(self.sapmles_per_layer[depth], len(neighbor_nodes))
            select_nodes = probas.multinomial(num_samples=sample_size, replacement=False).cpu()
            

            # XXX: with Concat workaround
            select_nodes = torch.LongTensor(np.unique(np.concatenate([np.arange(len(prev_nodes)), select_nodes])))
            after_nodes = neighbor_nodes[select_nodes]
            
            tmp_adj = current_layer_adj[:, after_nodes]
            prev_nodes = after_nodes

            nodeblock.layers_adj.insert(0, scipy_to_sptensor(tmp_adj))
            nodeblock.sampled_nodes.insert(0, prev_nodes)
            nodeblock.sampled_nodes_aux.insert(0, probas[select_nodes])

        nodeblock.sampled_nodes = np.array(nodeblock.sampled_nodes)
        nodeblock.layers_adj = np.array(nodeblock.layers_adj)
        nodeblock.sampled_nodes_aux = np.array(nodeblock.sampled_nodes_aux)

        return nodeblock