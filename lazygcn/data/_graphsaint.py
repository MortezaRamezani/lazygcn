import sys
import json
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler

from ..utils.helper import encode_onehot, create_mask


def load_graphsaint(self):

    adj_full = sp.load_npz(
        '{}/adj_full.npz'.format(self.base_dir)).astype(np.bool)
    adj_train = sp.load_npz(
        '{}/adj_train.npz'.format(self.base_dir)).astype(np.bool)
    role = json.load(open('{}/role.json'.format(self.base_dir)))
    features = np.load('{}/feats.npy'.format(self.base_dir))
    class_map = json.load(open('{}/class_map.json'.format(self.base_dir)))

    assert len(class_map) == features.shape[0]
    class_map = {int(k): v for k, v in class_map.items()}

    # find onehot label if multiclass or not
    num_vertices = adj_full.shape[0]
    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
        class_arr = np.zeros((num_vertices, num_classes))
        for k, v in class_map.items():
            class_arr[k] = v

        self.is_multiclass = True
        labels = class_arr
    else:
        num_classes = max(class_map.values()) - min(class_map.values()) + 1
        class_arr = np.zeros((num_vertices, num_classes))
        offset = min(class_map.values())
        for k, v in class_map.items():
            class_arr[k][v-offset] = 1
        labels = np.where(class_arr)[1]
    
    # ---- normalize feats ----
    # train_nodes = np.array(list(set(adj_train.nonzero()[0])))
    # train_feats = features[train_nodes]
    # scaler = StandardScaler()
    # scaler.fit(train_feats)
    # features = scaler.transform(features)
    # -------------------------


    self.graph.adj = adj_full.astype(float)
    self.graph.num_nodes = num_vertices
    self.graph.num_edges = adj_full.nnz

    self.features = features
    self.num_features = features.shape[1]

    self.labels = labels
    self.num_classes = num_classes

    self.train_mask = create_mask(role['tr'], num_vertices)
    self.val_mask = create_mask(role['va'], num_vertices)
    self.test_mask = create_mask(role['te'], num_vertices)
