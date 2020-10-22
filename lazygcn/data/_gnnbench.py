import os
import sys
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp

from ..utils.helper import encode_onehot, create_mask


def load_gnnbench(self):

    with np.load("{}/{}.npz".format(self.base_dir, self.name)) as f:

        features = sp.csr_matrix((f['attr_data'], f['attr_indices'], f['attr_indptr']),
                      f['attr_shape']).todense()

        features[features > 0] = 1

        adj = sp.csr_matrix((f['adj_data'], f['adj_indices'], f['adj_indptr']),
                            f['adj_shape'])
        labels = f['labels']

        # build symmetric adjacency matrix
        self.graph.adj = adj + \
            adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        self.graph.num_nodes = features.shape[0]
        self.graph.num_edges = self.graph.adj.nnz

        # default split for plantoid
        num_train = self.graph.num_nodes - 1500
        num_val = 500

        self.features = features
        self.num_features = features.shape[1]

        self.labels = labels
        self.num_classes = np.unique(labels).shape[0]

        self.train_mask = create_mask(range(num_train), labels.shape[0])
        self.val_mask = create_mask(
            range(num_train, num_train + num_val), self.labels.shape[0])
        self.test_mask = create_mask(
            range(num_train + num_val, self.graph.num_nodes), labels.shape[0])
