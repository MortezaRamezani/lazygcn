import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from collections import defaultdict
# from tqdm.notebook import trange, tqdm
# from .. import profiler

from ..utils.helper import is_notebook


class DefaultMLP(object):

    def __init__(self,
                 config,
                 dataset,
                 model,
                 minibatch,
                 optimizer=torch.optim.Adam,
                 loss_func=nn.CrossEntropyLoss,
                 activation=F.relu):

        self.config = config
        self.dataset = dataset

        # set device
        if self.config.gpu >= 0:
            self.device = self.config.gpu
        else:
            self.device = 'cpu'

        # model info
        self.model = model(config.num_layers,
                           dataset.num_features,
                           config.hidden_dim,
                           dataset.num_classes,
                           activation,
                           dropout=config.dropout,
                           weight_decay=config.weight_decay).to(self.device)

        self.optimizer = optimizer(self.model.parameters(),
                                   lr=config.lr,
                                   weight_decay=config.weight_decay)

        self.loss_func = loss_func()

        if self.dataset.is_multiclass:
            self.loss_func = nn.BCEWithLogitsLoss()

        self.minibatch = minibatch(
            dataset.train_index, config.train_batch_size, config.supervised)

        self.wall_clock = []
        self.train_loss = []
        self.val_loss = []
        self.val_score = []
        self.test_score = []


        self.epoch_wall_clock = []
        self.iter_train_loss = []
        self.grad_vars = []
        self.timing_hist = defaultdict(list)

        self.best_model = None
        self.best_val_score = 0
        self.best_val_iter = 0
        self.train_time = 0
        self.total_time = 0

        # profiler.profiler_config.flush_stats()

        if is_notebook():
            from tqdm.notebook import trange, tqdm
        else:
            from tqdm import tqdm, trange

        self.trange = trange

    def run(self):

        iter_cnt = 0

        val_score = 0
        val_loss = 0

        tbar = self.trange(self.config.num_iters, desc='Training Iterations')
        for itr in tbar:

            iter_start = time.perf_counter()

            batch = self.minibatch()

            batch_inputs, batch_labels = self.dataset.load_batch_data(
                batch, batch, self.device)

            # forward pass on the model
            self.model.train()
            pred = self.model(batch_inputs)

            # compute loss
            loss = self.loss_func(pred, batch_labels)
            iter_loss = loss.item()

            # do backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            iter_end = time.perf_counter()
            self.wall_clock.append(iter_end - iter_start)
            self.train_loss.append(iter_loss)

            # validation
            if not self.config.no_val and iter_cnt > 0 and iter_cnt % self.config.val_frequency == 0:
                self.model.eval()
                val_loss, val_score = self.validation()
                self.val_loss.append(val_loss)
                self.val_score.append(val_score)

            tbar.set_description('training iteration #{}'.format(itr+1))
            tbar.set_postfix(loss=loss.item(), val_score=val_score)

            # to capture start of next iteration, including sampling
            iter_cnt += 1
            iter_start = time.perf_counter()

        print('iter {}, loss:{:.4f}, val-loss:{:.4f}, val-score:{:.4f}'.format(itr,
                                                                               loss.item(),
                                                                               val_loss, val_score))

    def validation(self):

        batch = self.dataset.train_index
        # print(batch)
        batch_inputs, batch_labels = self.dataset.load_batch_data(
            batch, batch, self.device)

        pred = self.model(batch_inputs)
        loss = self.loss_func(pred, batch_labels)

        if self.dataset.is_multiclass:
            pred_labels = (pred.detach().cpu() > 0)
        else:
            pred_labels = pred.detach().cpu().argmax(dim=1)

        score = f1_score(batch_labels.cpu(), pred_labels, average="micro")

        return loss.item(), score


class RecyclingMLP(DefaultMLP):
    def __init__(self, *args):
        super().__init__(*args)
        self.num_iters = self.config.num_iters
        self.max_rec_iters = self.config.recycle_period

    def run(self):
        iter_cnt = 0
        val_score = 0
        val_loss = 0

        tbar = trange(self.num_iters, desc='Training Iterations')
        for itr in tbar:
            iter_start = time.perf_counter()

            batch = self.minibatch()

            batch_inputs, batch_labels = self.dataset.load_batch_data(
                batch, batch, self.device)

            for _ in range(self.max_rec_iters):
                self.model.train()

                pred = self.model(batch_inputs)

                # compute loss
                loss = self.loss_func(pred, batch_labels)
                iter_loss = loss.item()

                # do backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                iter_end = time.perf_counter()
                self.wall_clock.append(iter_end - iter_start)
                self.train_loss.append(iter_loss)

                # validation
                if iter_cnt > 0 and iter_cnt % self.config.val_frequency == 0:
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

        print('epoch {}, loss:{:.4f}, val-loss:{:.4f}, val-score:{:.4f}'.format(itr,
                                                                                loss.item(), val_loss, val_score))
