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


class FullGCN(BaseGCN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):

        iter_cnt = 10
        iter_loss = 0

        tbar = self.trange(self.config.num_epochs, desc='Training Iterations')

        nb = self.val_nodeblock
        # Inputs for full is same as validation
        batch_inputs = self.val_inputs
        # All training nodes labels for full-batch
        batch_labels = self.dataset.labels[self.dataset.train_index].pin_memory()
        batch_labels = batch_labels.to(self.device, non_blocking=True)

        for epoch in tbar:
            
            pred = self.model(nb, batch_inputs, full=True)

            if type(pred) == list:
                pred = pred[-1]

            pred = pred[self.dataset.train_index]

            loss = self.loss_func(pred, batch_labels)
            iter_loss = loss.item()

            # do backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_score = self.calc_f1(pred, batch_labels)

            self.train_loss.append(iter_loss)

            # Validation
            val_loss, val_score = self.do_validation(iter_cnt)

            tbar.set_description(
                'training iteration #{}'.format(iter_cnt))
            tbar.set_postfix(loss=iter_loss, epoch_score=np.mean(epoch_score), val_score=val_score)

        self.do_inference()