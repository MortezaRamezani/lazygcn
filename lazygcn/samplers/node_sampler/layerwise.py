import numpy as np

from ...graph import NodeBlock
from ...utils.helper import scipy_to_sptensor


class LayerWiseNodeSampler(object):

    def __init__(self, graph, num_layers, sapmles_per_layer):
        self.graph = graph
        self.num_layers = num_layers
        self.sapmles_per_layer = sapmles_per_layer

    def __call__(self, seed):

        if callable(seed):
            seed = seed()

        nodeblock = NodeBlock()
        
        num_nodes = self.graph.num_nodes

        prev_nodes = seed
        nodeblock.sampled_nodes.insert(0, prev_nodes)
        
        for depth in range(self.num_layers):
            current_layer_adj = self.graph.adj[prev_nodes, :]

            neigh_index = np.unique(current_layer_adj.indices)
            num_samples = np.min([len(neigh_index), self.sapmles_per_layer[depth]])
            next_nodes = np.random.choice(num_nodes, num_samples, replace=False)
            next_nodes = np.unique(np.concatenate((next_nodes, prev_nodes)))

            nodeblock.layers_adj.insert(0,
                scipy_to_sptensor(current_layer_adj[:, next_nodes]))
            nodeblock.sampled_nodes.insert(0, next_nodes)

            prev_nodes = next_nodes


        nodeblock.sampled_nodes = np.array(nodeblock.sampled_nodes)
        nodeblock.layers_adj = np.array(nodeblock.layers_adj)
        
        return nodeblock