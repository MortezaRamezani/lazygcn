import torch
import numpy as np


from ...graph import NodeBlock
from ...utils.helper import scipy_to_sptensor, row_normalize, adj_to_indval


class LADIESNodeSampler(object):

    def __init__(self, graph, num_layers, sapmles_per_layer):
        self.graph = graph
        self.num_layers = num_layers
        self.sapmles_per_layer = sapmles_per_layer

        self.laplacian_sq = self.graph.adj.multiply(self.graph.adj)


    def __call__(self, seed, *args, **kwargs):

        if callable(seed):
            seed = seed()

        nodeblock = NodeBlock()

        num_nodes = self.graph.num_nodes

        prev_nodes = seed
        nodeblock.sampled_nodes.insert(0, prev_nodes)

        for depth in range(self.num_layers):
            current_layer_adj = self.graph.adj[prev_nodes, :]

            pi = np.array(np.sum(self.laplacian_sq[prev_nodes, :], axis=0))[0]
            p = pi / np.sum(pi)

            num_samples = np.min(
                [np.sum(p > 0), self.sapmles_per_layer[depth]])

            next_nodes = np.random.choice(
                num_nodes, num_samples, p=p, replace=False)

            # next_nodes = np.unique(np.concatenate((next_nodes, prev_nodes)))
            # next_nodes = np.unique(np.concatenate((next_nodes, seed)))
            
            # We need to keep prev_node at the begining of matrix for SAGENet
            next_nodes = np.unique(next_nodes)
            next_nodes = np.setdiff1d(next_nodes, prev_nodes)
            next_nodes = np.concatenate((prev_nodes, next_nodes))
        
            tmp_adj = row_normalize(current_layer_adj[:, next_nodes].multiply(1/p[next_nodes]))
            
            # Debug
            # tmp_adj = row_normalize(current_layer_adj[:, next_nodes])
            # print(tmp_adj.shape,  end=', ')


            nodeblock.layers_adj.insert(0, scipy_to_sptensor(tmp_adj))
            # nodeblock.layers_adj.insert(0, adj_to_indval(tmp_adj))
            # nodeblock.layers_adj.insert(0, tmp_adj)
            nodeblock.sampled_nodes.insert(0, next_nodes)

            prev_nodes = next_nodes

            # Debug for num_samp = 0
            # print(np.array_equiv(np.sort(prev_nodes,axis=0), np.sort(seed,axis=0)))
        
        # print()
            
        nodeblock.sampled_nodes = np.array(nodeblock.sampled_nodes)
        nodeblock.layers_adj = np.array(nodeblock.layers_adj)
        
        return nodeblock