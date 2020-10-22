
import time
import numpy as np
import torch
import torch.nn as nn

from tqdm.notebook import trange, tqdm

from .base_gcn import BaseGCN
from ..utils.helper import is_notebook
from .. import profiler
from ..graph import NodeBlock


class RecycleExpGCN(BaseGCN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.done_iter = 0
        self.rho = self.config.rho
        self.tau = self.config.recycle_period
        self.num_iters = self.config.num_iters

    def generate_taus(self, T):
        taus = []
        k = 0
        total_taus = 0
        while total_taus < T:
            tau_i = int(self.tau * np.power(self.rho, k))
            if tau_i == 0:
                tau_i = 1 
            if total_taus + tau_i > T:
                tau_i = T - total_taus
            taus.append(tau_i)
            total_taus += tau_i
            k += 1

        return taus

    def run(self):
        iter_cnt = 0
        val_score = 0

        total_iteration = self.config.num_epochs * self.config.num_iters

        tbar = self.trange(total_iteration, desc='Training Iterations')
        taus = self.generate_taus(total_iteration)

        ti = 0
        epoch_loss = []
        epoch_wall_clock = []

        iter_start = time.perf_counter()
        sampling_start = time.perf_counter()

        # ? Helpful for debugging
        # for nb in self.sampler.flash_sampling(5):
        #     nb.to_device(self.device)
        #     self.nb = nb
        #     self.ssnb = self._subsample_all(nb)

        with np.printoptions(threshold=10, linewidth=100, edgeitems=10):
            print(np.array(taus))
            print(np.sum(taus))


        if self.config.num_workers == -1:
            generator = range(len(taus))
        else:
            generator = self.sampler.flash_sampling(len(taus))

        # It's not fair comparison right now normal has more thread while validating
        # for nb in self.sampler.flash_sampling(len(taus)):
        # for _ in range(len(taus)):
        #     nb = self.sampler()

        per_sample_loss = []
        prev_minibatch = None

        for nb in generator:

            if self.config.num_workers == -1:
                nb = self.sampler(loss=per_sample_loss, minibatch=prev_minibatch)


            ############################# Sampling #############################
            input_nodes = nb.sampled_nodes[0]
            output_nodes = nb.sampled_nodes[-1]

            torch.cuda.synchronize()
            sampling_end = time.perf_counter()
            
            prev_minibatch = output_nodes
            
            ############################# Transfer #############################
            with profiler.Timer('transfer') as transfer_time:
                # move nb and corresponding features and labels to GPU
                batch_inputs, batch_labels = self.dataset.load_batch_data(
                    input_nodes, output_nodes, self.device, supervised=self.config.supervised)
                nb.to_device(self.device)
                torch.cuda.synchronize()

            for rec_itr in range(taus[ti]):

                recycle_vector = None
                new_nb = nb
                new_batch_inputs = batch_inputs
                new_batch_labels = batch_labels

                ######################## Recycling Overhead ########################
                if rec_itr != 0:
                    with profiler.Timer('recycle') as recycle_time:
                        if self.config.rec_subsample == 'outer':
                            # Fix inner, subsample outer
                            recycle_vector = torch.cuda.FloatTensor(
                                len(output_nodes)).uniform_() > 0.2
                            # recycle_vector = np.random.choice(
                            #     len(output_nodes), size=int(len(output_nodes) * 0.8), replace=False)
                            # recycle_vector = torch.cuda.LongTensor(recycle_vector)
                            new_batch_labels = batch_labels[recycle_vector]
                        elif self.config.rec_subsample == 'all':
                            # subsample inner and outer
                            new_nb = self._subsample_all(nb)
                            new_batch_inputs = batch_inputs[new_nb.sampled_nodes[0]]
                            new_batch_labels = batch_labels[new_nb.sampled_nodes[-1]]
                        elif self.config.rec_subsample == 'none':
                            # fix all
                            new_batch_labels = batch_labels

                        torch.cuda.synchronize()

                ############################# Compute #############################
                with profiler.Timer('compute') as compute_time:
                    self.model.train()

                    # pred = self.model(new_nb, new_batch_inputs, recycle_vector, full=nb.full)
                    pred = self.model(new_nb, new_batch_inputs, None, full=nb.full)
                    
                    # pred = pred[output_nodes]
                    if recycle_vector != None:
                        pred = pred[recycle_vector]

                    # compute loss                 
                    if self.config.minibatch_method == 'adaptloss' and rec_itr == 0:
                        loss_func_each = nn.CrossEntropyLoss(reduction='none')
                        loss_each = loss_func_each(pred, new_batch_labels)
                        per_sample_loss = loss_each.detach().cpu().numpy()
                        loss = torch.mean(loss_each)
                    else:
                        loss = self.loss_func(pred, new_batch_labels)
                    
                    iter_loss = loss.detach().item()

                    # do backward
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    torch.cuda.synchronize()

                iter_end = time.perf_counter()

                iter_score = self.calc_f1(pred, new_batch_labels)

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

                self.wall_clock.append(iter_end - iter_start)
                self.train_loss.append(iter_loss)

                # Validation
                val_loss, val_score = self.do_validation(iter_cnt, prev_score=val_score)

                if self.config.grad_var:
                    iter_gv = self.gradient_variance()
                    self.grad_vars.append(iter_gv)

                tbar.set_description('training iteration #{}'.format(iter_cnt+1))
                tbar.set_postfix(loss=iter_loss, train_score=iter_score, val_score=val_score)
                tbar.update(1)

                # to capture start of next iteration, including sampling
                iter_cnt += 1
                iter_start = time.perf_counter()
                sampling_start = time.perf_counter()

            
            ti += 1
        
        self.do_inference()


    def _subsample_all(self, nbs):
        ratio = 0.8
        nodeblock = NodeBlock()

        row, _ , _ = nbs.layers_adj[-1].coo()
        row = torch.unique(row)
        ss_row_index = torch.randperm(len(row))[:int(len(row) * ratio)]
        ss_row = row[ss_row_index]
        nodeblock.sampled_nodes.insert(0, ss_row)

        for layer in reversed(nbs.layers_adj):

            _, col, _ = layer.index_select(0, ss_row).coo()
            col = torch.unique(col)
            ss_col_index = torch.randperm(len(col))[:int(len(col) * ratio)]
            ss_col = col[ss_col_index]

            new_layer = layer.index_select(0, ss_row).index_select(1, ss_col)

            ss_row = ss_col

            nodeblock.layers_adj.insert(0, new_layer)
            nodeblock.sampled_nodes.insert(0, ss_row)
        
        return nodeblock