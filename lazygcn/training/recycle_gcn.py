
import time
import numpy as np
import torch

from tqdm.notebook import trange, tqdm

from .base_gcn import BaseGCN
from ..utils.helper import is_notebook
from .. import profiler


class RecycleGCN(BaseGCN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.done_iter = 0
        self.rho = self.config.rho
        self.tau = self.config.recycle_period
        self.num_iters = self.config.num_iters
        self.exp_rec = self.config.exp_rec

    # @property

    def num_fresh_iters(self, ep):
        self.done_iter = 0
        if not self.exp_rec:
            return int(np.ceil(self.num_iters / self.config.recycle_period))

    def num_rec_iters(self, ep):
        if not self.exp_rec:
            remaining_iter = (self.num_iters - self.done_iter)
            if  remaining_iter < self.config.recycle_period:
                self.done_iter += remaining_iter
                return remaining_iter
            else:
                self.done_iter += self.config.recycle_period
                return self.config.recycle_period

    def run(self):
        iter_cnt = 0
        val_score = 0

        tbar = self.trange(self.config.num_epochs, desc='Training Iterations')

        iter_start = time.perf_counter()
        sampling_start = time.perf_counter()

        for epoch in tbar:
            epoch_loss = []
            epoch_wall_clock = []
            iter_start = time.perf_counter()
            sampling_start = time.perf_counter()

            for nb in self.sampler.flash_sampling(self.num_fresh_iters(epoch)):

                ############################# Sampling #############################
                input_nodes = nb.sampled_nodes[0]
                output_nodes = nb.sampled_nodes[-1]

                torch.cuda.synchronize()
                sampling_end = time.perf_counter()

                ############################# Transfer #############################
                with profiler.Timer('transfer') as transfer_time:
                    # move nb and corresponding features and labels to GPU
                    batch_inputs, batch_labels = self.dataset.load_batch_data(
                        input_nodes, output_nodes, self.device, supervised=self.config.supervised)
                    nb.to_device(self.device)
                    torch.cuda.synchronize()

                for rec_itr in range(self.num_rec_iters(epoch)):

                    recycle_vector = None
                    new_batch_labels = batch_labels

                    ######################## Recycling Overhead ########################
                    if rec_itr != 0:
                        # ! do or don't np.unique?
                        with profiler.Timer('recycle') as recycle_time:
                            if self.config.rec_subsample:
                                recycle_vector = torch.cuda.FloatTensor(
                                    len(output_nodes)).uniform_() > 0.2
                                new_batch_labels = batch_labels[recycle_vector]
                            else:
                                new_batch_labels = batch_labels
                            torch.cuda.synchronize()

                    ############################# Compute #############################
                    with profiler.Timer('compute') as compute_time:
                        self.model.train()

                        pred = self.model(nb, batch_inputs, recycle_vector, full=nb.full)

                        # compute loss
                        loss = self.loss_func(pred, new_batch_labels)
                        iter_loss = loss.item()

                        # do backward
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        torch.cuda.synchronize()

                    iter_end = time.perf_counter()

                    ############################### Logging ###############################
                    if rec_itr == 0:
                        self.timing_hist['sampling'].append(
                            sampling_end - sampling_start)
                        self.timing_hist['transfer'].append(
                            transfer_time.elapsed)
                    else:
                        self.timing_hist['recycle'].append(
                            recycle_time.elapsed)
                    self.timing_hist['compute'].append(compute_time.elapsed)

                    epoch_loss.append(iter_loss)
                    self.wall_clock.append(iter_end - iter_start)
                    epoch_wall_clock.append(iter_end - iter_start)
                    self.iter_train_loss.append(iter_loss)

                    if self.config.grad_var:
                        iter_gv = self.gradient_variance()
                        self.grad_vars.append(iter_gv)

                    # to capture start of next iteration, including sampling
                    iter_cnt += 1
                    iter_start = time.perf_counter()
                    sampling_start = time.perf_counter()

            epoch_loss = np.mean(epoch_loss)
            self.train_loss.append(epoch_loss)
            self.epoch_wall_clock.append(np.sum(epoch_wall_clock))

            # Validation
            val_loss, val_score = self.do_validation(iter_cnt)

            tbar.set_description(
                'training iteration #{}'.format(iter_cnt+1))
            tbar.set_postfix(loss=epoch_loss, val_score=val_score)
        
        self.do_inference()