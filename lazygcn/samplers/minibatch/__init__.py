
import torch
import time
import numpy as np


class MiniBatch(object):
    def __init__(self, seed_nodes, batch_size, supervised=False):
        self.seed_nodes = seed_nodes
        self.batch_size = batch_size
        self.supervised = supervised

        # * Uneccesary
        # np.random.seed(int(str(time.perf_counter()).split('.')[1]))

    def __iter__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass


class FullBatch(MiniBatch):
    def __init__(self, seed_nodes, batch_size, supervised):
        super().__init__(seed_nodes, len(seed_nodes), supervised)

    def __call__(self, *args, **kwargs):
        return self.seed_nodes


class SplitBatch(MiniBatch):
    def __init__(self, seed_nodes, batch_size, supervised):
        super().__init__(seed_nodes, batch_size, supervised)
        self.seed_nodes = seed_nodes
        self.batch_size = batch_size

        seed_nodes = torch.LongTensor(self.seed_nodes)
        self.batches = torch.split(seed_nodes, self.batch_size)

    def __iter__(self, *args, **kwargs):
        seed_nodes = torch.LongTensor(self.seed_nodes)
        for batch in torch.split(seed_nodes, self.batch_size):
            yield batch.numpy()

    def __len__(self):
        return len(self.batches)

    def __call__(self, bid, *args, **kwargs):
        return self.batches[bid]

class RandomBatch(MiniBatch):
    def __init__(self, seed_nodes, batch_size, supervised):
        super().__init__(seed_nodes, batch_size, supervised)
        self.num_batches = int(np.ceil(len(self.seed_nodes) / self.batch_size))

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = np.random.choice(
                self.seed_nodes, self.batch_size, replace=False)
            yield batch

    def __call__(self, *args, **kwargs):
        # To support unsupervised where training nodes are sorted
        if not self.supervised:
            batch = np.random.choice(
                self.seed_nodes, self.batch_size, replace=False)
        else:
            batch = np.random.choice(
                len(self.seed_nodes), self.batch_size, replace=False)
        
        # return batch
        return np.sort(batch)




__all__ = ['MiniBatch', 
           'SplitBatch', 
           'RandomBatch', 
           'FullBatch']
