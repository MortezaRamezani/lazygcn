import os
import sys
import scipy.sparse as sp
import numpy as np


def load_reddit(self):

    # * fix for reddit self loop
    self.name = 'reddit_self_loop'
    self.base_dir = self._get_dataset_dir()
    
    # graph
    coo_adj = sp.load_npz(os.path.join(self.base_dir,
                                       "{}_graph.npz".format(self.name)))
    # features and labels
    reddit_data = np.load(os.path.join(self.base_dir, "reddit_data.npz"))

    self.graph.adj = coo_adj
    self.features = reddit_data["feature"]
    self.labels = reddit_data["label"]

    self.num_features = self.features.shape[1]
    self.num_classes = np.unique(self.labels).shape[0]

    self.graph.num_nodes = self.features.shape[0]
    self.graph.num_edges = self.graph.adj.nnz

    # tarin/val/test indices
    node_ids = reddit_data["node_ids"]
    node_types = reddit_data["node_types"]

    self.train_mask = (node_types == 1)
    self.val_mask = (node_types == 2)
    self.test_mask = (node_types == 3)
