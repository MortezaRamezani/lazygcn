import os
import sys
import time
import numpy as np
import scipy
import scipy.sparse as sp
import torch
import copy
import networkx as nx
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler

from ..graph import Graph
from ..utils.helper import row_normalize, create_mask


class Dataset(object):

    #@profile('load')
    def __init__(self, dataset, sym=False, tvt=None):

        self.name = dataset
        self.base_dir = self._get_dataset_dir()

        self.graph = Graph()
        self.features = []

        self.labels = []
        self.num_classes = []
        self.num_features = []

        # TODO: these should be BoolTensor, array of 1 and 0 won't work for masking!!
        self.train_mask = []
        self.val_mask = []
        self.test_mask = []

        self.is_multiclass = False
        self.sym = sym
        
        self._load_data()
        self._preprocess()

        if tvt is not None:
            # change train/val/test ratio 
            n = self.graph.num_nodes
            tr = int(tvt[0] * n / 100)
            vr = int(tvt[1] * n / 100)
            idx_train = range(tr)
            idx_val = range(tr, tr+vr)
            idx_test = range(tr+vr, n)
            self.train_mask = create_mask(idx_train, n)
            self.val_mask = create_mask(idx_val, n)
            self.test_mask = create_mask(idx_test, n)
        

        self.num_train = self.train_mask.sum()
        self.num_val = self.val_mask.sum()
        self.num_test = self.test_mask.sum()

        self.train_index = self.train_mask.nonzero()[0]
        self.val_index = self.val_mask.nonzero()[0]
        self.test_index = self.test_mask.nonzero()[0]


    from ._cora import load_cora
    from ._citation import load_citation
    from ._graphsaint import load_graphsaint
    from ._reddit import load_reddit
    from ._gnnbench import load_gnnbench

    def remove_missing(self, missing='pairnorm', missing_rate=0):
        
        if missing == 'random':
            self.features = torch.empty(self.features.shape).random_(2)
        
        elif missing == 'pairnorm':
            # use 0 for test/val features
            train_mask = torch.BoolTensor(self.train_mask)
            erasing_seeds = np.arange(self.graph.num_nodes)[~train_mask]
            size = int(len(erasing_seeds) * (missing_rate/100))
            erased_nodes = np.random.choice(erasing_seeds, size=size, replace=False)
            self.features[erased_nodes] = 0
    
    def remove_edges(self, missing_rate=0):
        adj = self.graph.adj.tocoo()
        nnz = adj.nnz
        perm = np.random.permutation(nnz)
        preserve_nnz = int(nnz*missing_rate)
        perm = perm[:preserve_nnz]
        adj_matrix = sp.csr_matrix((adj.data[perm],
                                    (adj.row[perm], adj.col[perm])),
                                    shape=adj.shape)
        self.graph.adj = adj_matrix

    def sparsify(self, spanner=1):
        print('before', self.graph.adj.nnz)
        adj = copy.deepcopy(self.graph.adj)
        adj[adj>0] = 1
        nx_graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.Graph)
        sparse_graph = nx.spanner(nx_graph, spanner)
        self.graph.adj = nx.adjacency_matrix(sparse_graph)
        print('after', self.graph.adj.nnz)
        self.graph.normalize_adj(symetric=self.sym)


    def change_split(self, tvt):
        n = self.graph.num_nodes
        tr = int(tvt[0] * n / 100)
        vr = int(tvt[1] * n / 100)
        idx_train = range(tr)
        idx_val = range(tr, tr+vr)
        idx_test = range(tr+vr, n)
        self.train_mask = create_mask(idx_train, n)
        self.val_mask = create_mask(idx_val, n)
        self.test_mask = create_mask(idx_test, n)

        self.num_train = self.train_mask.sum()
        self.num_val = self.val_mask.sum()
        self.num_test = self.test_mask.sum()

        self.train_index = self.train_mask.nonzero()[0]
        self.val_index = self.val_mask.nonzero()[0]
        self.test_index = self.test_mask.nonzero()[0]

    
    def change_split_index(self, train_start, train_end, val_start, val_end, test_start, test_end):
        n = self.graph.num_nodes
        # tr = int(tvt[0] * n / 100)
        # vr = int(tvt[1] * n / 100)
        idx_train = range(train_start, train_end)
        idx_val = range(val_start, val_end)
        idx_test = range(test_start, test_end)
        self.train_mask = create_mask(idx_train, n)
        self.val_mask = create_mask(idx_val, n)
        self.test_mask = create_mask(idx_test, n)

        self.num_train = self.train_mask.sum()
        self.num_val = self.val_mask.sum()
        self.num_test = self.test_mask.sum()

        self.train_index = self.train_mask.nonzero()[0]
        self.val_index = self.val_mask.nonzero()[0]
        self.test_index = self.test_mask.nonzero()[0]

    def load_batch_data(self, input_nodes, output_nodes, device, supervised=False):

        non_blocking=False

        if supervised:
            input_nodes = self.train_index[input_nodes]
            output_nodes = self.train_index[output_nodes]

        if type(input_nodes) == str and input_nodes == 'all':
            batch_inputs = self.features #.pin_memory()
        else:
            batch_inputs = self.features[input_nodes] #.pin_memory()
        batch_inputs = batch_inputs.to(device, non_blocking=non_blocking)
        
        if type(output_nodes) == str and output_nodes == 'all':
            batch_labels = self.labels #.pin_memory()
        else:
            batch_labels = self.labels[output_nodes] #.pin_memory()
        batch_labels = batch_labels.to(device, non_blocking=non_blocking)

        return batch_inputs, batch_labels

    def _load_data(self):
        if self.name == 'cora':
            self.load_cora()
        elif self.name in ['citeseer', 'pubmed']:
            self.load_citation()
        elif self.name in ['flickr', 'yelp', 'ppi', 'ppi-large', 'reddit', 'amazon']:
            self.load_graphsaint()
        elif self.name in ['coauthorcs']:
            self.load_gnnbench()
        else:
            raise NotImplementedError

    def _preprocess(self):
        # TODO: set multilabel flag

        self.graph.normalize_adj(symetric=self.sym)
        self._preprocess_feat()

        # Why not do it here?
        self.features = torch.FloatTensor(self.features)
        if self.is_multiclass:
            self.labels = torch.FloatTensor(self.labels)
        else:
            self.labels = torch.LongTensor(self.labels)

    def _get_dataset_dir(self):
        default_dir = os.path.join(os.path.expanduser('~'), '.gnn')
        dirname = os.environ.get('GNN_DATASET_DIR', default_dir)
        return os.path.join(dirname, self.name)

    def _preprocess_feat(self):

        # if self.name == 'yelp':
        
        train_nodes = self.train_mask.nonzero()[0]
        train_feats = self.features[train_nodes]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        self.features = scaler.transform(self.features)

        # # features must be sparse scipy
        
        # else:
        # features = row_normalize(self.features)
        # if type(features) == scipy.sparse.lil.lil_matrix \
        #         or type(features) == scipy.sparse.csr.csr_matrix:
        #     features = np.array(features.todense())
        # else:
        #     features = np.array(features)
        # self.features = features

    def __str__(self):
        param = ['#Nodes', '#Edges', '#Classes', '#Features',
                 '#Train samples', '#Val samples', '#Test samples']
        value = [self.graph.num_nodes,
                 self.graph.num_edges,
                 self.num_classes,
                 self.features.shape[1],
                 self.num_train,
                 self.num_val,
                 self.num_test]
        tab_value = [[param[i], value[i]] for i in range(len(param))]
        return tabulate(tab_value, headers=['Param', 'Value'], floatfmt=".0f")
