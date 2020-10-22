import numpy as np

from ...graph import NodeBlock
from ...utils.helper import scipy_to_sptensor


class FullGraph(object):

    def __init__(self, graph, num_layers, *args):
        self.graph = graph
        self.num_layers = num_layers

    def __call__(self, seed):

        if callable(seed):
            seed = seed()
            
        nodeblock = NodeBlock()
        nodeblock.full = True
        nodeblock.sampled_nodes = [seed]
        nodeblock.layers_adj = [scipy_to_sptensor(self.graph.adj)]

        nodeblock.sampled_nodes = np.array(nodeblock.sampled_nodes)
        nodeblock.layers_adj = np.array(nodeblock.layers_adj)


        return nodeblock