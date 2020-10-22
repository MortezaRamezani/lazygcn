import time
import numpy as np
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from sklearn.metrics import f1_score

from .base_gcn import BaseGCN
from ..graph import NodeBlock
from .. import samplers
from .. import profiler

import multiprocessing as mp


class SpiderVRGCN(BaseGCN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epoch = np.ceil(self.config.num_iters /
                                 len(self.sampler)).astype(int)

    def run(self):

        iter_cnt = 0
        iter_loss = 0
        val_score = 0

        tbar = self.trange(self.config.num_epochs *
                           self.config.num_iters, desc='Training Iterations')

        full_graph = NodeBlock(self.dataset.graph.adj)
        full_graph.to_device(self.device)

        full_features = self.dataset.features.to(
            self.device, non_blocking=True)
        full_train_labels = self.dataset.labels[self.dataset.train_index].to(
            self.device, non_blocking=True)

        for _ in range(self.config.num_epochs):
            iter_start = time.perf_counter()

            self.model.train()

            pre_full = copy.deepcopy(self.model)
            pre_mini = copy.deepcopy(self.model)

            # compute full grad
            self.optimizer.zero_grad()
            full_pred = self.model(full_graph, full_features, full=True)
            full_loss = self.loss_func(
                full_pred[self.dataset.train_index], full_train_labels)
            full_loss.backward()
            self.optimizer.step()

            epoch_loss = []
            for nb in self.sampler.flash_sampling(self.config.num_iters):

                # Sampling
                input_nodes = nb.sampled_nodes[0]
                output_nodes = nb.sampled_nodes[-1]

                # Transfer
                nb.to_device(self.device)
                batch_inputs, batch_labels = self.dataset.load_batch_data(input_nodes,
                                                                          output_nodes,
                                                                          self.device)
                ################### Compute ###################

                # pre_full.grad <- net.grad
                for full_par, net_par in zip(pre_full.parameters(), self.model.parameters()):
                    full_par.grad = copy.deepcopy(net_par.grad)

                # pre_mini.partial_grad()
                pre_mini.zero_grad()
                mini_pred = pre_mini(nb, batch_inputs)
                mini_loss = self.loss_func(mini_pred, batch_labels)
                mini_loss.backward()

                # net.partial_grad()
                self.optimizer.zero_grad()
                pred = self.model(nb, batch_inputs)
                loss = self.loss_func(pred, batch_labels)
                loss.backward()
                iter_loss = loss.item()

                # net.grad <- net.grad + pre_full.grad - pre_mini.grad
                for net_par, full_par, mini_par in zip(self.model.parameters(), pre_full.parameters(), pre_mini.parameters()):
                    net_par.grad.data = net_par.grad.data + full_par.grad.data - mini_par.grad.data
                    
                # pre_mini.weights <- net.weights
                for mini_par, net_par in zip(pre_mini.parameters(), self.model.parameters()):
                    mini_par.data = copy.deepcopy(net_par.data)

                # update net weights
                self.optimizer.step()

                # torch.cuda.synchronize()
                iter_end = time.perf_counter()

                epoch_loss.append(iter_loss)

                if self.config.grad_var:
                    iter_gv = self.gradient_variance()
                    self.grad_vars.append(iter_gv)

                self.wall_clock.append(iter_end - iter_start)
                # self.train_loss.append(iter_loss)

                # to capture start of next iteration, including sampling
                iter_cnt += 1
                iter_start = time.perf_counter()

            self.train_loss.append(np.mean(epoch_loss))

            # Validation
            val_loss, val_score = self.do_validation(iter_cnt)

            tbar.set_description(
                'training iteration #{}'.format(iter_cnt+1))
            tbar.set_postfix(loss=np.mean(epoch_loss), val_score=val_score)
            tbar.update(self.config.num_iters)

    def full_gradients(self):
        pass
