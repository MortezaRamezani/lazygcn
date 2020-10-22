import os
import sys
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp

from ..utils.helper import encode_onehot, create_mask


def load_cora(self, num_train=1208, num_val=500):
    features_file = "{}/cora.content".format(self.base_dir)
    edges_file = "{}/cora.cites".format(self.base_dir)

    idx_features_labels = np.genfromtxt(features_file, dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(edges_file, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]),
                         (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    self.graph.adj = adj + \
        adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    self.graph.num_nodes = features.shape[0]
    self.graph.num_edges = self.graph.adj.nnz

    self.features = features.todense()
    self.num_features = features.shape[1]

    self.labels = np.where(labels)[1]
    self.num_classes = labels.shape[1]

    self.train_mask = create_mask(range(num_train), labels.shape[0])
    self.val_mask = create_mask(
        range(num_train, num_train + num_val), self.labels.shape[0])
    self.test_mask = create_mask(
        range(num_train + num_val, self.graph.num_nodes), labels.shape[0])
