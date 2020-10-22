import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from sklearn.metrics import f1_score

from .base_gcn import BaseGCN
from ..utils.helper import is_notebook, load_tqdm

from .. import profiler

import multiprocessing as mp


class NormalGCN(BaseGCN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):

        iter_cnt = 0
        iter_loss = 0
        val_score = 0

        tbar = self.trange(self.config.num_epochs, desc='Training Iterations')

        for epoch in tbar:
            iter_start = time.perf_counter()
            sampling_start = time.perf_counter()
            
            epoch_loss = []
            epoch_score = []
            epoch_wall_clock = []


            if self.config.num_workers == -1:
                generator = range(self.config.num_iters)
            else:
                generator = self.sampler.flash_sampling(self.config.num_iters)
            
            # for nb in self.sampler.flash_sampling(self.config.num_iters):

            for nb in generator:

                if self.config.num_workers == -1:
                    nb = self.sampler()

                # Sampling
                input_nodes = nb.sampled_nodes[0]
                output_nodes = nb.sampled_nodes[-1]

                torch.cuda.synchronize()
                sampling_end = time.perf_counter()

                # Transfer
                transfer_start = time.perf_counter()
                nb.to_device(self.device)
                batch_inputs, batch_labels = self.dataset.load_batch_data(input_nodes,
                                                                          output_nodes,
                                                                          self.device,
                                                                          supervised=self.config.supervised)
                torch.cuda.synchronize()
                transfer_end = time.perf_counter()

                # Compute
                compute_start = time.perf_counter()

                iter_loss, iter_score = self.train_step(nb, batch_inputs, batch_labels, output_nodes)

                torch.cuda.synchronize()
                iter_end = time.perf_counter()

                epoch_loss.append(iter_loss)
                epoch_score.append(iter_score)

                self.wall_clock.append(iter_end - iter_start)
                epoch_wall_clock.append(iter_end - iter_start)
                self.iter_train_loss.append(iter_loss)

                self.timing_hist['sampling'].append(
                    sampling_end - sampling_start)
                self.timing_hist['transfer'].append(
                    transfer_end - transfer_start)
                self.timing_hist['compute'].append(iter_end - compute_start)

                if self.config.grad_var:
                    iter_gv = self.gradient_variance()
                    self.grad_vars.append(iter_gv)

                # to capture start of next iteration, including sampling
                iter_cnt += 1
                sampling_start = time.perf_counter()
                iter_start = time.perf_counter()

            epoch_loss = np.mean(epoch_loss)
            self.train_loss.append(epoch_loss)
            self.epoch_wall_clock.append(np.sum(epoch_wall_clock))

            # Validation
            val_loss, val_score = self.do_validation(epoch, prev_score=val_score)

            tbar.set_description(
                'training iteration #{}'.format(iter_cnt))
            tbar.set_postfix(loss=epoch_loss, epoch_score=np.mean(epoch_score), val_score=val_score)

        self.do_inference()