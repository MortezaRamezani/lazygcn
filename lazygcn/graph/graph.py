from ..utils.helper import row_normalize
import scipy.sparse as sp
import numpy as np

class Graph(object):
    def __init__(self):
        self.adj = []
        self.num_nodes = 0
        self.num_edges = 0

    def normalize_adj(self, symetric=False):
        
        self.adj = self.adj.tolil()
        self.adj.setdiag(1)
        
        # self.adj = row_normalize(self.adj)
        
        deg = None
        sort_indices = True

        if symetric:
            print('sym')
            """Normalization by D^{-1/2} (A+I) D^{-1/2}."""
            rowsum = np.array(self.adj.sum(1)) + 1e-20
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt, 0)
            adj_norm = self.adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        
        else:
            diag_shape = (self.adj.shape[0], self.adj.shape[1])
            D = self.adj.sum(1).flatten() if deg is None else deg
            norm_diag = sp.dia_matrix((1/D,0),shape=diag_shape)
            adj_norm = norm_diag.dot(self.adj)
            if sort_indices:
                adj_norm.sort_indices()

        self.adj = adj_norm