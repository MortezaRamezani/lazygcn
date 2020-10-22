
from ..utils.helper import scipy_to_sptensor


class NodeBlock(object):
    """
    NodeBlock includes adjacency matrix and required features and labels for current epoch of training
    """

    def __init__(self, graph=None):
        self.layers_adj = []
        self.sampled_nodes = []
        self.sampled_nodes_aux = []
        self.subgraph_mapping = []
        self.full = False

        if graph is not None:
            self.full_graph(graph)

    def __getitem__(self, layer_id):
        return self.layers_adj[layer_id]

    def to_device(self, device='cpu', aux=False):
        """ 
        move current layer object to device
        """

        non_blocking = False

        for i, adj in enumerate(self.layers_adj):
            # * Maybe add non_blocking=True
            self.layers_adj[i] = adj.to(device, non_blocking=non_blocking)

        if aux:
            for i, layer_aux in enumerate(self.sampled_nodes_aux):
                self.sampled_nodes_aux[i] = layer_aux.to(
                    device, non_blocking=non_blocking)

        # ? To support spmm, but apparently it's using too much memory
        # for i, adj in enumerate(self.layers_adj):
        #     self.layers_adj[i][0] = adj[0].to(device)
        #     self.layers_adj[i][1] = adj[1].to(device)
        #     # don't send size to CUDA

    def full_graph(self, spmx):
        self.layers_adj = [scipy_to_sptensor(spmx)]
        # TODO: fill sampled nodes for all nodes
        self.sampled_nodes = []
        self.full = True
