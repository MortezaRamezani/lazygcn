import numpy as np

from ...graph import NodeBlock
from ...utils.helper import scipy_to_sptensor, row_normalize


class SubGraphSampler(object):

    def __init__(self, graph, num_layers, sapmles_per_layer, *args, **kwargs):
        self.graph = graph
        self.num_layers = num_layers
        self.sapmles_per_layer = sapmles_per_layer
        self.supervised = kwargs['supervised']

    def __call__(self, seed, *args, **kwargs):

        if callable(seed):
            seed = seed()

        nodeblock = NodeBlock()

        if self.supervised:
            subgraph_nodes = seed
        else:
            # ! not working :((
            nodeblock.sampled_nodes.insert(0, seed)
            # get all neighbors that are not training nodes and add them to subgraph
            curr_adj = self.graph.adj[seed, :]
            subgraph_nodes = np.unique(np.concatenate((seed, curr_adj.indices)))
            # we need the index of seed nodes in the subgraph nodes ( |V| > subgraph_nodes > seed)
            # https://stackoverflow.com/questions/33678543/finding-indices-of-matches-of-one-array-in-another-array
            nodeblock.subgraph_mapping = np.nonzero(np.in1d(subgraph_nodes, seed))[0]

        tmp_adj = self.graph.adj[subgraph_nodes, :][:, subgraph_nodes]
        # print(tmp_adj.shape)
        curr_adj = row_normalize(tmp_adj)

        nodeblock.full = True
        nodeblock.layers_adj.insert(0, scipy_to_sptensor(curr_adj))
        
        # don't add non-training nodes to sampled_nodes, used in gradient
        nodeblock.sampled_nodes.insert(0, subgraph_nodes)

        nodeblock.sampled_nodes = np.array(nodeblock.sampled_nodes)
        nodeblock.layers_adj = np.array(nodeblock.layers_adj)
        
        return nodeblock