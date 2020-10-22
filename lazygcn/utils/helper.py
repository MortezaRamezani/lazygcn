import sys
import torch
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from torch_sparse import spmm, SparseTensor


def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    with np.errstate(divide='ignore'):
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def encode_onehot(labels):
    classes = list(sorted(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def fast_onehot(labels):
    # TODO: won't work if labels aren't from 0 to l-1
    labels_onehot = np.zeros((labels.size, labels.max()+1))
    labels_onehot[np.arange(labels.size), labels] = 1
    return labels_onehot


def create_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return mask


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def pickle_load(pkl_file):
    if sys.version_info > (3, 0):
        return pkl.load(pkl_file, encoding='latin1')
    else:
        return pkl.load(pkl_file)


def scipy_to_sptensor(scsp_matrix):
    sparse_mx = scsp_matrix.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)

    # return indices, values, shape
    #
    # return torch.sparse.FloatTensor(indices, values, shape)
    #
    # ! CUDA won't allow for Multi-process to execute pin_memory operation
    return SparseTensor(row=indices[0], col=indices[1],
                        value=values, sparse_sizes=shape)

    # pin_memory for faster transfer to cuda
    # return SparseTensor(row=indices[0], col=indices[1],
    #                     value=values, sparse_sizes=shape).pin_memory()



def adj_to_indval(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)

    return [indices, values, shape]
    # return torch.sparse.FloatTensor(indices, values, shape)


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def load_tqdm():
    if is_notebook():
        from tqdm.notebook import trange, tqdm
    else:
        from tqdm import tqdm, trange


def capture_activations(layer, inputs, outputs):
    setattr(layer, "activations", inputs[0].detach())


def capture_backprops(layer, inputs, outputs):
    setattr(layer, "backprops", outputs[0].detach())


def calculate_sample_grad(layer):
    A = layer.activations
    B = layer.backprops

    n = A.shape[0]
    B = B * n
    weight_grad = torch.einsum('ni,nj->nij', B, A)
    bias_grad = B
    grad_norm = torch.sqrt(weight_grad.norm(p=2, dim=(1, 2)).pow(
        2) + bias_grad.norm(p=2, dim=1).pow(2)).squeeze().detach()
    return grad_norm  # , weight_grad


"""
make the figure smooth
"""
def smooth(scalars, weight=0.8):  # Weight between 0 and 1
    if len(scalars) == 0:
        return scalars
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed