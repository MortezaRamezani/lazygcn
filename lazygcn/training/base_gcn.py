import os
import time
import copy
import yaml
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from sklearn.metrics import f1_score
# from tqdm.notebook import trange, tqdm

from ..utils.helper import is_notebook
from .. import samplers
from .. import profiler
from ..graph import NodeBlock


class BaseGCN(object):

    def __init__(self,
                 config,
                 dataset,
                 model,
                 sampler,
                 optimizer=torch.optim.Adam,
                 loss_func=nn.CrossEntropyLoss,
                 activation=F.relu):

        self.config = copy.deepcopy(config)
        self.dataset = dataset

        # Set num iterations
        if self.config.num_iters == -1:
            self.config.num_iters = int(np.ceil(
                len(self.dataset.train_index) / self.config.train_batch_size))
            print('Number of iterations per epoch:', self.config.num_iters)

        # set device
        if self.config.gpu >= 0:
            self.device = self.config.gpu
        else:
            self.device = 'cpu'

        if self.config.cpu_val:
            self.val_device = 'cpu'
        else:
            self.val_device = self.device

        # model info
        self.model = model(config.num_layers,
                           dataset.num_features,
                           config.hidden_dim,
                           dataset.num_classes,
                           activation,
                           dropout=config.dropout,
                           weight_decay=config.weight_decay,
                           concat=config.concat).to(self.device)

        # print(self.model)

        self.optimizer = optimizer(self.model.parameters(),
                                   lr=config.lr,
                                   weight_decay=config.weight_decay)

        self.loss_func = loss_func()

        if self.dataset.is_multiclass:
            self.loss_func = nn.BCEWithLogitsLoss()
            

        self.sampler = sampler(dataset.graph,
                               dataset.train_index,
                               config.num_layers,
                               config.train_batch_size,
                               config.num_workers,
                               config.samples_per_layer,
                               config.supervised,
                               config.minibatch_method,
                               config.sampling_method,
                               dataset=dataset)  # for dpp

        self.val_sampler = samplers.Sampler(dataset.graph,
                                            dataset.val_index,
                                            config.num_layers,
                                            config.val_batch_size,
                                            minibatch_method='split',
                                            sampling_method='exact')

        self.test_sampler = samplers.Sampler(self.dataset.graph,
                                            self.dataset.test_index,
                                            self.config.num_layers,
                                            self.config.val_batch_size,
                                            minibatch_method='split',
                                            sampling_method='exact')

        self.wall_clock = []
        self.epoch_wall_clock = []
        self.train_loss = []
        self.iter_train_loss = []
        self.val_loss = []
        self.val_score = []
        self.test_score = []
        self.grad_vars = []
        self.timing_hist = defaultdict(list)

        self.best_model = None
        self.best_val_score = 0
        self.best_val_iter = 0
        self.train_time = 0
        self.total_time = 0

        # in order to save time on validation and since it's exact
        if not self.config.no_val and not self.config.minibatch_val :
            # ! Let's keep full graph for validation, not ideal but we can! other than Amazon
            self.val_nodeblock = NodeBlock(self.dataset.graph.adj)
            self.val_nodeblock.to_device(self.val_device)
            self.val_inputs, self.val_labels = self.dataset.load_batch_data(
                'all', self.dataset.val_index, self.val_device)
        
        if self.config.minibatch_val:
            self.val_sampler.prepare_full()
            self.val_nodeblocks = []
                

        if self.config.grad_var:
            # Load Full Graph and Features
            self.full_graph = NodeBlock(self.dataset.graph.adj)
            self.full_graph.to_device(self.device)
            self.full_features = self.dataset.features.to(
                self.device, non_blocking=True)
            self.full_train_labels = self.dataset.labels[self.dataset.train_index].to(
                self.device, non_blocking=True)

        if is_notebook():
            from tqdm.notebook import trange, tqdm
        else:
            from tqdm import tqdm, trange

        self.trange = trange

        # if self.config.log:

        setattr(self.config, 'dataset', self.dataset.name)
            

    @property
    def run_id(self):    
        prefix = '{}_{}_{}_'.format(
            self.dataset.name, 
            self.model.__class__.__name__.lower(),
            self.config.postfix
            )

        run_id = 1
        if os.path.exists(self.config.log_dir):
            for fn in os.listdir(self.config.log_dir):
                if fn.startswith(prefix) and fn.endswith('.model'):
                    run_id += 1
        else:
            pass
            # print(prefix + '{:03d}'.format(run_id))

        return prefix + '{:03d}'.format(run_id)


    @property
    def key(self):
        pass

    @property
    def meta(self):
        meta = {'train': self.__class__.__name__,
                'model': self.model.__class__.__name__,
                'minibatch': self.config.minibatch_method,
                'sampling': self.config.sampling_method,
                'batchsize': self.config.train_batch_size,
                'numworkers': self.config.num_workers,
                }
        return meta

    def run(self):
        # TODO: replace this with pytorch dataloader
        # TODO: this is sequential sampling/training with one iteration per epochs
        # TODO: use with caution

        iter_cnt = 0
        val_score = 0
        iter_loss = 0
        
        if self.config.sampling_method == 'full':
            
            if self.config.supervised:
                nb = self.sampler()
                nb.to_device(self.device)
                batch_inputs, batch_labels = self.dataset.load_batch_data(
                self.dataset.train_index, self.dataset.train_index, self.device)
                output_nodes = torch.arange(len(self.dataset.train_index))
            else:
                # save space use full graph from validation
                nb = self.val_nodeblock
                output_nodes = self.dataset.train_index
                batch_inputs = self.val_inputs
                batch_labels = self.dataset.labels[self.dataset.train_index].pin_memory()
                batch_labels = batch_labels.to(self.device, non_blocking=True)
            

        tbar = self.trange(self.config.num_epochs, desc='Training Iterations')

        for itr in tbar:
            iter_start = time.perf_counter()

            with profiler.Timer('sampling') as sampling_time:
                if self.config.sampling_method != 'full':
                    nb = self.sampler()
                    input_nodes = nb.sampled_nodes[0]
                    output_nodes = nb.sampled_nodes[-1]

            with profiler.Timer('transfer') as transfer_time:
                # move nb and corresponding features and labels to GPU
                if self.config.sampling_method != 'full':
                    nb.to_device(self.device)
                    batch_inputs, batch_labels = self.dataset.load_batch_data(
                        input_nodes, output_nodes, self.device)

            with profiler.Timer('compute') as compute_time:
                iter_loss, iter_score = self.train_step(nb, batch_inputs, batch_labels, output_nodes)

            iter_end = time.perf_counter()

            self.wall_clock.append(iter_end - iter_start)
            self.train_loss.append(iter_loss)

            # validation
            val_loss, val_score = self.do_validation(iter_cnt, val_score)

            tbar.set_description('training iteration #{}'.format(itr))
            tbar.set_postfix(loss=iter_loss, val_score=val_score)

            self.timing_hist['sampling'].append(sampling_time.elapsed)
            self.timing_hist['transfer'].append(transfer_time.elapsed)
            self.timing_hist['compute'].append(compute_time.elapsed)

            iter_cnt += 1

        self.do_inference()

    def train_step(self, nb, batch_inputs, batch_labels, output_nodes):
        # forward pass on the model
        self.model.train()

        pred = self.model(nb, batch_inputs, full=nb.full)
        
        # To support multi-layer output
        if type(pred) == list:
                pred = pred[-1]

        # necessary for somecases (supervised/full, ...)
        if self.sampler.is_subgraph and not self.config.supervised:
            print(nb.layers_adj[0].sizes())
            pred = pred[nb.subgraph_mapping]
        elif self.config.sampling_method == 'full':
            pred = pred[output_nodes]

        # compute loss
        loss = self.loss_func(pred, batch_labels)
        iter_loss = loss.item()

        # do backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        iter_score = self.calc_f1(pred, batch_labels)

        return iter_loss, iter_score

    def do_validation(self, iter_cnt, prev_score=None):
        if not self.config.no_val and iter_cnt > 0 and iter_cnt % self.config.val_frequency == 0:
            self.model.eval()
            val_loss, val_score = self.validation()
            self.val_loss.append(val_loss)
            self.val_score.append(val_score)

            if val_score > self.best_val_score:
                self.best_val_score = val_score
                self.best_val_iter = iter_cnt
                self.best_model = copy.deepcopy(self.model)

            # print(val_loss)
            return val_loss, val_score
            
        else:
            return 0, prev_score
    
    def do_inference(self):
        # ignore first iterations time due to pytorch overhead
        self.total_time = np.sum(self.wall_clock[1:])
        self.train_time = np.sum(self.wall_clock[1:self.best_val_iter])
        _, test_score = self.inference()
        self.test_score = test_score
        print('Total training time: {:.2f}'.format(self.total_time))
        print('Total training time till best validation {:.2f}'.format(self.train_time))
        print('Test Score on Best Model: {:.2f}% at iteration {}'.format(test_score * 100, self.best_val_iter))
        print('Log filename', self.run_id)


    def calc_f1(self, pred, batch_labels):
        if self.dataset.is_multiclass:
            # pred_labels = (pred.detach().cpu() > 0)
            
            pred_labels = nn.Sigmoid()(pred.detach().cpu())
            pred_labels[pred_labels > 0.5] = 1
            pred_labels[pred_labels <= 0.5] = 0

            # pred_labels = pred.detach().cpu()
            # pred_labels[pred_labels >= 0] = 1
            # pred_labels[pred_labels < 0] = 0
        else:
            pred_labels = pred.detach().cpu().argmax(dim=1)

        score = f1_score(batch_labels.cpu(), pred_labels, average="micro")

        return score


    def validation(self):

        if not self.config.minibatch_val:

            nb = self.val_nodeblock
            batch_inputs = self.val_inputs
            batch_labels = self.val_labels
            
            if self.val_device == self.device:
                pred = self.model(nb, batch_inputs, full=True)
            else:
                val_model = copy.deepcopy(self.model)
                val_model.to(self.val_device)
                val_model.eval()
                pred = val_model(nb, batch_inputs, full=True)
            
            if type(pred) == list:
                pred = pred[-1]
            pred = pred[self.dataset.val_index]

            loss = self.loss_func(pred, batch_labels)
            val_loss = loss.item()
            val_score = self.calc_f1(pred, batch_labels)
        
        else:
        # if True:
            val_loss = []
            val_score = []

            if len(self.val_nodeblocks) == 0:
                print('Waiting for validation minibatch...')
                self.val_nodeblocks = self.val_sampler.get_full()
                print('Validation minibatch is ready...')

            for nb in self.val_nodeblocks:
                self.goh = nb
                val_batch_inputs_nodes = nb.sampled_nodes[0]
                val_batch_outputs_nodes = nb.sampled_nodes[-1]
                nb.to_device(self.device)
                val_batch_inputs, val_batch_labels = self.dataset.load_batch_data(
                    val_batch_inputs_nodes, val_batch_outputs_nodes, self.device
                )

                pred = self.model(nb, val_batch_inputs, full=False)
            
                if type(pred) == list:
                    pred = pred[-1]

                loss = self.loss_func(pred, val_batch_labels)
                score = self.calc_f1(pred, val_batch_labels)

                val_batch_size = len(val_batch_outputs_nodes)
                val_loss.append(loss.item() * val_batch_size)
                val_score.append(score * val_batch_size)

            val_loss = np.sum(val_loss) / self.dataset.num_val
            val_score = np.sum(val_score) / self.dataset.num_val


        return val_loss, val_score

    def inference(self):

        if not self.config.minibatch_val:
            # test nodeblock same as validation (full-graph)
            nb = self.val_nodeblock

            # same as validation, all nodes
            batch_inputs = self.val_inputs

            # load test labels to device
            batch_labels = self.dataset.labels[self.dataset.test_index] #.pin_memory()
            batch_labels = batch_labels.to(self.val_device, non_blocking=True)

            if self.best_model is None:
                self.best_model = self.model

            self.best_model.to(self.val_device)
                
            pred = self.best_model(nb, batch_inputs, full=True)
            if type(pred) == list:
                pred = pred[-1]
            pred = pred[self.dataset.test_index]

            loss = self.loss_func(pred, batch_labels)
            test_loss = loss.item()
            test_score = self.calc_f1(pred, batch_labels)
        
        else:

            print('Waiting for test minibatch...')
            self.test_sampler.prepare_full()
            self.test_nodeblocks = self.test_sampler.get_full()
            print('Test minibatch is ready...')

            test_loss = []
            test_score = []

            for nb in self.test_nodeblocks:
                test_batch_inputs_nodes = nb.sampled_nodes[0]
                test_batch_outputs_nodes = nb.sampled_nodes[-1]
                nb.to_device(self.device)
                test_batch_inputs, test_batch_labels = self.dataset.load_batch_data(
                    test_batch_inputs_nodes, test_batch_outputs_nodes, self.device
                )

                if self.best_model is None:
                    self.best_model = self.model

                pred = self.best_model(nb, test_batch_inputs, full=False)
            
                if type(pred) == list:
                    pred = pred[-1]

                loss = self.loss_func(pred, test_batch_labels)
                score = self.calc_f1(pred, test_batch_labels)

                test_batch_size = len(test_batch_outputs_nodes)
                test_loss.append(loss.item() * test_batch_size)
                test_score.append(score * test_batch_size)

            test_loss = np.sum(test_loss) / self.dataset.num_test
            test_score = np.sum(test_score) / self.dataset.num_test

        return test_loss, test_score

    
    def per_layer_score(self):
        nb = self.val_nodeblock

        # same as validation, all nodes
        batch_inputs = self.val_inputs

        # load test labels to device
        batch_labels = self.dataset.labels[self.dataset.test_index] #.pin_memory()
        batch_labels = batch_labels.to(self.val_device, non_blocking=True)

        if self.best_model is None:
            self.best_model = self.model

        self.best_model.to(self.val_device)
            
        pred = self.best_model(nb, batch_inputs, full=True)
        
        self.per_layer_test = []
        for p in pred:
            plp = p[self.dataset.test_index]
            test_score = self.calc_f1(plp, batch_labels)
            self.per_layer_test.append(test_score)
    
    def gradient_variance(self):

        net_grads = []

        for p_net in self.model.parameters():
            net_grads.append(p_net.grad.data)

        clone_net = copy.deepcopy(self.model)

        pred = clone_net(self.full_graph, self.full_features, full=True)
        loss = self.loss_func(
            pred[self.dataset.train_index], self.full_train_labels)
        loss.backward()

        clone_net_grad = []
        for p_net in clone_net.parameters():
            clone_net_grad.append(p_net.grad.data)
        del clone_net

        variance = 0.0
        for g1, g2 in zip(net_grads, clone_net_grad):
            variance += (g1-g2).norm(2) ** 2
        variance = torch.sqrt(variance)
        return variance.cpu()



    def save_results(self):
        result = {'wall_clock': self.wall_clock,
                  'epoch_wallclock': self.epoch_wall_clock,
                  'train_loss': self.train_loss,
                  'iter_train_loss': self.iter_train_loss,
                  'val_loss': self.val_loss,
                  'val_score': self.val_score,
                  'test_score': self.test_score,
                  'grad_vars': self.grad_vars,
                  'timing_hist': self.timing_hist,
                  'best_val_it': self.best_val_iter,
                  'total_time': self.total_time,
                  'train_time': self.train_time}

        np.save(self.log_path+'.result', result)

    def save_model(self):
        torch.save(self.best_model.state_dict(), self.log_path + '.model')

    def save_config(self):
        all_vars = copy.deepcopy(vars(self.config))
        all_vars.pop('samples_per_layer')
        all_vars.pop('prefix')
        all_vars['trainer'] = self.__class__.__name__

        with open(self.log_path + '.yaml', 'w') as outfile:
            yaml.dump(all_vars, outfile)

    def save(self):
        self.make_logdir()
        self.log_path = os.path.join(self.config.log_dir, self.run_id)
        print('saving to ...:', self.log_path)
        self.save_model()
        self.save_config()
        self.save_results()

    def make_logdir(self):
        if self.config.log and not os.path.exists(self.config.log_dir):
            os.makedirs(self.config.log_dir)