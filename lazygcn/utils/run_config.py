
import os
import argparse
import numpy as np
import datetime


class RunConfig(object):

    def __init__(self, desc='Default GCN Config', args=[]):

        self.parser = argparse.ArgumentParser(description=desc)

        self.parser.add_argument('--model', type=str, required=False, default='gcn',
                                 help='The model')

        self.parser.add_argument('--concat', type=bool, default=False,
                                 help='Should self and aggregated be concated together?')

        # Dataset params
        self.parser.add_argument('--dataset', type=str, required=False, default='cora',
                                 help='The input dataset')

        # Learning params
        self.parser.add_argument('--num-iters', type=int, default=100,
                                 help='number of training epochs')
        self.parser.add_argument('--num-epochs', type=int, default=100,
                                 help='number of training epochs')
        self.parser.add_argument('--num-layers', type=int, default=3,
                                 help='number of hidden gcn layers')
        self.parser.add_argument('--hidden-dim', type=int, default=64,
                                 help='number of hidden gcn units')
        self.parser.add_argument('--lr', type=float, default=1e-2,
                                 help='learning rate')
        self.parser.add_argument('--dropout', type=float, default=0,
                                 help='dropout probability')
        self.parser.add_argument('--weight-decay', type=float, default=0,
                                 help='Weight for L2 loss')

        # Batching specific params
        self.parser.add_argument('--num-workers', type=int, default=10,
                                 help='Number of sampler workers')
        self.parser.add_argument('--train-batch-size', type=int, default=512,
                                 help='batch size')
        self.parser.add_argument('--test-batch-size', type=int, default=512,
                                 help='test batch size')
        self.parser.add_argument('--train-sample-size', type=int, default=256,
                                 help='number of neighbors to be sampled')
        self.parser.add_argument('--test-sample-size', type=int, default=64,
                                 help='number of neighbors to be sampled')

        self.parser.add_argument('--minibatch-method', type=str, required=False, default='split',
                                 help='The minibatch')
        self.parser.add_argument('--sampling-method', type=str, required=False, default='exact',
                                 help='Node sampling methods')

        self.parser.add_argument('--supervised', type=bool, default=False,
                                 help='Use all nodes feature for training or not')

        # Recycle Setting
        self.parser.add_argument('--recycle-period', type=int, default=5,
                                 help='number of iteration to recycle the samples')
        self.parser.add_argument('--rho', type=int, default=1,
                                 help='rho')
        self.parser.add_argument('--exp-rec', type=bool, default=False,
                                 help='Should use expo recyling')
        self.parser.add_argument('--rec-subsample', type=str, default='outer',
                                 help='Subsample when recycling?')

        # GPU Settings
        self.parser.add_argument('--gpu', type=int, default=0,
                                 help='gpu')

        # Test settings
        self.parser.add_argument('--no-val', type=bool, default=False,
                                 help='No validation')
        self.parser.add_argument('--cpu-val', type=bool, default=False,
                                 help='Validation on CPU')
        self.parser.add_argument('--minibatch-val', type=bool, default=False,
                                 help='Mini batch validation')
        self.parser.add_argument('--val-batch-size', type=int, default=1000,
                                 help='Validation batch size')
        self.parser.add_argument('--grad-var', type=bool, default=False,
                                 help='Compute Gradients Variance')
        self.parser.add_argument('--val-frequency', type=int, default=10,
                                 help='How often do the test during training')
        self.parser.add_argument('--test-frequency', type=int, default=5,
                                 help='How often do the test during training')
        self.parser.add_argument('--stop-threshold', type=int, default=0.01,
                                 help='Validation threshold to stop trainging')

        self.parser.add_argument('--log', type=bool, default=False,
                                 help='prefix for logging result/model')
        self.parser.add_argument('--prefix', type=str, default='',
                                 help='prefix for logging result/model')
        self.parser.add_argument('--postfix', type=str, default='',
                                 help='postfix for logging result/model')

        self.config = self.parser.parse_args(args=args)

        self._set_samples_layer()

        log_folder = '{}/{}'.format(self.config.prefix,
                                        datetime.datetime.now().strftime("%m%d-%H%M%S"))
        setattr(self.config, 'log_dir', log_folder)
            

    def update_args(self, new_args={}):
        for key, value in new_args.items():
            setattr(self.config, key, value)
        self._set_samples_layer()

        if self.config.log:
            print('log dir:', self.config.log_dir)

    def _set_samples_layer(self):
        # ! no additional layers
        setattr(self.config, 'samples_per_layer', np.array([self.config.train_sample_size]
                                                           * self.config.num_layers))

    def load_yaml(self):
        pass
