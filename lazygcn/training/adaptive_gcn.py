import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from sklearn.metrics import f1_score

from .base_gcn import BaseGCN
from ..utils.helper import is_notebook, load_tqdm
from torch_sparse import spmm

from .. import profiler

import multiprocessing as mp


class AdaptiveGCN(BaseGCN):
    def __init__(self,
                 config,
                 dataset,
                 model,
                 sampler,
                 optimizer=torch.optim.Adam,
                 loss_func=nn.CrossEntropyLoss,
                 activation=F.relu):
        super().__init__(config,
                         dataset,
                         model,
                         sampler,
                         optimizer,
                         loss_func,
                         activation)

        # init weights
        self.sample_weights = nn.Parameter(torch.randn(
            (dataset.num_features, 1), dtype=torch.float32))
        nn.init.xavier_uniform_(self.sample_weights)

        params = list(self.model.parameters())
        params.append(self.sample_weights)
        self.optimizer = optimizer(params,
                                   lr=config.lr,
                                   weight_decay=config.weight_decay)

        self.lamb = 0.5

    def run(self):
        iter_cnt = 0
        iter_loss = 0

        tbar = self.trange(self.config.num_epochs, desc='Training Iterations')

        for epoch in tbar:
            epoch_loss = []
            for _ in range(self.config.num_iters):

                iter_start = time.perf_counter()

                nb = self.sampler(sample_weights=self.sample_weights)

                input_nodes = nb.sampled_nodes[0]
                output_nodes = nb.sampled_nodes[-1]

                # Transfer
                nb.to_device(self.device, aux=True)
                batch_inputs, batch_labels = self.dataset.load_batch_data(input_nodes,
                                                                          output_nodes,
                                                                          self.device)

                # forward pass on the model
                self.model.train()
                pred, var_loss = self.model(nb, batch_inputs, full=nb.full)

                # compute loss
                loss = self.loss_func(pred, batch_labels)

                total_loss = loss + self.lamb * var_loss
                iter_loss = total_loss.item()

                # do backward
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                iter_end = time.perf_counter()

                epoch_loss.append(iter_loss)
                iter_cnt += 1

            epoch_loss = np.mean(epoch_loss)
            self.train_loss.append(epoch_loss)
            self.wall_clock.append(iter_end - iter_start)

            # Validation
            val_loss, val_score = self.do_validation(epoch)

            tbar.set_description(
                'training iteration #{}'.format(iter_cnt))
            tbar.set_postfix(loss=epoch_loss, val_score=val_score)
        
        self.do_inference()
