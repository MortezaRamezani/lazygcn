import numpy as np

from ...graph import NodeBlock
from ...utils.helper import scipy_to_sptensor, row_normalize


class NodeWiseNodeSampler(object):

    def __init__(self, graph, num_layers, sapmles_per_layer):
        self.graph = graph
        self.num_layers = num_layers
        self.sapmles_per_layer = sapmles_per_layer

    def __call__(self, seed, *args, **kwargs):

        if callable(seed):
            seed = seed()

        nodeblock = NodeBlock()

        prev_nodes = seed
        nodeblock.sampled_nodes.insert(0, prev_nodes)
        
        for depth in range(self.num_layers):
            
            current_layer_adj = self.graph.adj[prev_nodes, :]
            
            # next_nodes = [prev_nodes]
            next_nodes = []
            
            row_start_stop = np.lib.stride_tricks.as_strided(current_layer_adj.indptr, 
                                                            shape=(current_layer_adj.shape[0], 2),
                                                            strides=2*current_layer_adj.indptr.strides)
            for start, stop in row_start_stop:  
                neigh_index  = current_layer_adj.indices[start:stop]
                num_samples = np.min([neigh_index.size, self.sapmles_per_layer[depth]])
                sampled_nodes = np.random.choice(neigh_index, num_samples, replace=False)
                next_nodes.append(sampled_nodes)

            next_nodes = np.unique(np.concatenate(next_nodes))
            next_nodes = np.setdiff1d(next_nodes, prev_nodes)
            next_nodes = np.concatenate((prev_nodes, next_nodes))
        
                
            # ! Super slow and acquire GIL in Multi-threads
            # for row in current_layer_adj:
            #     neigh_index = row.indices
            #     num_samples = np.min([neigh_index.size, self.sapmles_per_layer[depth]])
            #     sampled_nodes = np.random.choice(neigh_index, num_samples, replace=False)
            #     next_nodes = np.unique(np.concatenate((next_nodes, sampled_nodes)))

            tmp_adj = row_normalize(current_layer_adj[:, next_nodes])
            nodeblock.layers_adj.insert(0, scipy_to_sptensor(tmp_adj))
            nodeblock.sampled_nodes.insert(0, next_nodes)

            prev_nodes = next_nodes


        nodeblock.sampled_nodes = np.array(nodeblock.sampled_nodes)
        nodeblock.layers_adj = np.array(nodeblock.layers_adj)
        
        return nodeblock