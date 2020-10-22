import numpy as np

from ...graph import NodeBlock
from ...utils.helper import scipy_to_sptensor, adj_to_indval, row_normalize

class ExactNodeSampler(object):

    def __init__(self, graph, num_layers, *args):
        self.graph = graph
        self.num_layers = num_layers

    def __call__(self, seed):

        if callable(seed):
            seed = seed()
            
        nodeblock = NodeBlock()
        
        # ? Can we do better?
        prev_nodes = seed
        nodeblock.sampled_nodes.insert(0, prev_nodes)
        for _ in range(self.num_layers):
            current_layer_adj = self.graph.adj[prev_nodes, :]

            neigh_index = np.unique(current_layer_adj.indices)
            next_nodes = np.unique(np.concatenate((prev_nodes, neigh_index)))

            tmp_adj = current_layer_adj[:, next_nodes]
            # tmp_adj = row_normalize(tmp_adj)
            nodeblock.layers_adj.insert(0, scipy_to_sptensor(tmp_adj))
        
            nodeblock.sampled_nodes.insert(0, next_nodes)

            prev_nodes = next_nodes


        nodeblock.sampled_nodes = np.array(nodeblock.sampled_nodes)
        nodeblock.layers_adj = np.array(nodeblock.layers_adj)

        return nodeblock
