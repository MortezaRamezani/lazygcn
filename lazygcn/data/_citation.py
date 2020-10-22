import os
import sys
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp

from ..utils.helper import encode_onehot, create_mask, pickle_load, parse_index_file


def load_citation(self, num_train=-1, num_val=500):
    """
    Acquired from DGL

    Loads input data from gcn/data directory
    ind.name.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.name.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.name.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.name.x) as scipy.sparse.csr.csr_matrix object;
    ind.name.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.name.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.name.ally => the labels for instances in ind.name.allx as numpy.ndarray object;
    ind.name.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.name.test.index => the indices of test instances in graph, for the inductive setting as list object.
    All objects above must be saved using python pickle module.
    :param name: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """

    if num_train == -1:
        if self.name == 'pubmed':
            num_train = 18217
        elif self.name == 'citeseer':
            num_train = 1812

    root = self.base_dir
    objnames = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(objnames)):
        with open("{}/ind.{}.{}".format(root, self.name, objnames[i]), 'rb') as f:
            objects.append(pickle_load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "{}/ind.{}.test.index".format(root, self.name))
    test_idx_range = np.sort(test_idx_reorder)

    if self.name == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    onehot_labels = np.vstack((ally, ty))
    onehot_labels[test_idx_reorder, :] = onehot_labels[test_idx_range, :]
    labels = np.argmax(onehot_labels, 1)

    row_ind = [k for k, v in graph.items() for _ in range(len(v))]
    col_ind = [i for ids in graph.values() for i in ids]

    num_nodes = features.shape[0]
    num_edges = len(row_ind)
    adj = sp.coo_matrix((np.ones(num_edges), (row_ind, col_ind)),
                        shape=(num_nodes, num_nodes),
                        dtype=np.float32)

    idx_test = test_idx_range.tolist()
    idx_train = range(num_train)
    idx_val = range(num_train, num_train + num_val)
    
    # tr = 300
    # idx_train = range(tr)
    # idx_val = range(tr, tr+500)
    # idx_test = range(tr+500, num_nodes)

    self.graph.adj = adj
    self.graph.num_nodes = num_nodes
    self.graph.num_edges = num_edges

    self.features = features.todense()
    self.num_features = features.shape[1]
    
    self.labels = labels
    self.num_classes = onehot_labels.shape[1]

    self.train_mask = create_mask(idx_train, labels.shape[0])
    self.val_mask = create_mask(idx_val, labels.shape[0])
    self.test_mask = create_mask(idx_test, labels.shape[0])
