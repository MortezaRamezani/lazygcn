import time
import numpy as np
import torch

from tqdm.notebook import trange, tqdm

from .base_gcn import BaseGCN
from .. import profiler

class BaseRecycleGCN(BaseGCN):
    def __init__(self, *args):
        super().__init__(*args)

        # self.num_iters = np.ceil(self.config.num_iters /
        #                     self.config.recycle_period).astype(int)
        self.num_iters = self.config.num_iters

        self.max_rec_iters = self.config.recycle_period
        self.exp_rec = self.config.exp_rec


    # @property
    def num_rec_iters(self, it):
        if self.exp_rec:
            epoch_size = np.ceil(self.dataset.num_train / self.config.train_batch_size)
            new_num_rec_iters = np.power(2, int(it/epoch_size))
            if new_num_rec_iters <= self.max_rec_iters:
                return new_num_rec_iters
            else:
                return self.max_rec_iters
        else:
            return self.max_rec_iters

    def run(self):
        iter_cnt = 0
        val_score = 0
        val_loss = 0

        if self.config.exp_rec:
            num_epoch = self.config.num_iters
        else:
            num_epoch = np.ceil(self.config.num_iters / self.config.recycle_period).astype(int)

        tbar = trange(num_epoch, desc='Training Iterations')
        
        for itr in tbar:
            iter_start = time.perf_counter()

            with profiler.Timer('sampling') as sampling_time:

                nb = self.sampler()

                input_nodes = nb.sampled_nodes[0]
                output_nodes = nb.sampled_nodes[-1]

            with profiler.Timer('transfer') as transfer_time:
                # move nb and corresponding features and labels to GPU
                nb.to_device(self.device)
                batch_inputs, batch_labels = self.dataset.load_batch_data(
                    input_nodes, output_nodes, self.device)

            for rec_itr in range(self.num_rec_iters(iter_cnt)):
                
                recycle_vector = None
                new_batch_labels = batch_labels

                if rec_itr != 0:
                    # ! do or don't np.unique?
                    with profiler.Timer('recycle') as recycle_time:
                        recycle_vector = np.unique(np.random.choice(
                            len(output_nodes), size=len(output_nodes), replace=True))
                        recycle_vector = torch.LongTensor(recycle_vector)
                        new_batch_labels = batch_labels[recycle_vector]

                    with profiler.Timer('rectransfer') as rectran_time:
                        recycle_vector = recycle_vector.to(self.device)
                
                with profiler.Timer('compute') as compute_time:
                    self.model.train()

                    pred = self.model(nb, batch_inputs, recycle_vector)

                    # compute loss
                    loss = self.loss_func(pred, new_batch_labels)
                    iter_loss = loss.item()

                    # do backward
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                iter_end = time.perf_counter()

                if iter_cnt >= 1:
                    if rec_itr == 0:
                        self.timing_hist['sampling'].append(sampling_time.elapsed)
                        self.timing_hist['transfer'].append(transfer_time.elapsed)
                    else:
                        self.timing_hist['recycle'].append(recycle_time.elapsed)
                        self.timing_hist['rectran'].append(rectran_time.elapsed)

                    self.timing_hist['compute'].append(compute_time.elapsed)

                self.wall_clock.append(iter_end - iter_start)
                self.train_loss.append(iter_loss)

                # validation
                if not self.config.no_val and iter_cnt > 0 and iter_cnt % self.config.val_frequency == 0:
                    self.model.eval()
                    val_loss, val_score = self.validation()
                    self.val_loss.append(val_loss)
                    self.val_score.append(val_score)

                tbar.set_description(
                    'training iteration #{}'.format(iter_cnt+1))
                tbar.set_postfix(loss=loss.item(), val_score=val_score)

                # to capture start of next iteration, including sampling
                iter_cnt += 1
                iter_start = time.perf_counter()

            if iter_cnt > self.config.num_iters:
                break

        print('epoch {}, loss:{:.4f}, val-loss:{:.4f}, val-score:{:.4f}'.format(itr,
                                                                                loss.item(), val_loss, val_score))
