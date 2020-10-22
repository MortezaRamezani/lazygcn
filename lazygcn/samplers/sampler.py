
import torch
import numpy as np
import multiprocessing as mp
import concurrent.futures

from .minibatch import SplitBatch, RandomBatch, FullBatch
from .node_sampler import ExactNodeSampler, LayerWiseNodeSampler,\
    FullGraph, LADIESNodeSampler, NodeWiseNodeSampler

from ..graph import Graph


class Sampler(object):

    def __init__(self,
                 graph,
                 seed_nodes,
                 num_layers,
                 batch_size,
                 num_workers=1,
                 samples_per_layer=[],
                 supervised=False,
                 minibatch_method=None,
                 sampling_method=None,
                 **kwargs):

        self.num_nodes = len(seed_nodes)
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.is_subgraph = False

        extra_kw = {}
        if minibatch_method == None or minibatch_method == 'random':
            minibatch = RandomBatch
        elif minibatch_method == 'split':
            minibatch = SplitBatch
        elif minibatch_method == 'permute':
            minibatch = None
        elif minibatch_method == 'full':
            minibatch = FullBatch
        else:
            raise NotImplementedError

        self.minibatch = minibatch(seed_nodes, batch_size, supervised, **extra_kw)

        extra_kw = {}
        if sampling_method == None or sampling_method == 'exact':
            node_sampler = ExactNodeSampler
        elif sampling_method == 'layerwise':
            node_sampler = LayerWiseNodeSampler
        elif sampling_method == 'nodewise':
            node_sampler = NodeWiseNodeSampler
        elif sampling_method == 'ladies':
            node_sampler = LADIESNodeSampler
        elif sampling_method == 'full':
            node_sampler = FullGraph
        elif sampling_method == 'subgraph':
            self.is_subgraph = True
            node_sampler = SubGraphSampler
            extra_kw['supervised'] = supervised
        elif sampling_method == 'adaptive':
            node_sampler = AdaptiveNodeSampler
            extra_kw['features'] = kwargs['dataset'].features
        else:
            raise NotImplementedError

        if supervised:
            # only use training nodes
            train_graph = Graph()
            train_nodes = kwargs['dataset'].train_index
            train_graph.adj = graph.adj[train_nodes, :][:, train_nodes]
            train_graph.num_nodes = len(train_nodes)
            train_graph.num_edges = train_graph.adj.nnz
            graph = train_graph

        self.node_sampler = node_sampler(
            graph, num_layers, samples_per_layer, **extra_kw)
        self.num_workers = num_workers

    def __len__(self):
        # return int(self.num_nodes / self.batch_size)
        # to make it more efficent run num_workers at a time
        # return self.num_workers
        return len(self.minibatch)


    def __iter__(self):
        # TODO: accomodate for different batching (permute, random, dpp, split)
        for seed in self.minibatch:
            # return nodeblock for current seed
            yield self.node_sampler(seed)

    def __call__(self, *args, **kwargs):
        return self.node_sampler(self.minibatch(*args, **kwargs), *args, **kwargs)

    def prepare_full(self):

        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
            self.sampling_jobs = {executor.submit(
                self.node_sampler, self.minibatch(bid)): bid for bid in range(len(self.minibatch))}

    def get_full(self):
        # Order won't be the same, doesn't matter
        full_nodeblocks = []

        for future in concurrent.futures.as_completed(self.sampling_jobs):
            full_nodeblocks.append(future.result())
        
        return full_nodeblocks

    def flash_sampling(self, num_iter):
        max_threads = 20
        num_loop = int(np.ceil(num_iter/max_threads))
        num_iter_pt = max_threads
        
        for l in range(num_loop):

            if l == num_loop - 1:
                num_iter_pt = num_iter - ((num_loop - 1) * max_threads)
            
            # with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:

                # ? which one is better?
                sampling_jobs = {executor.submit(
                    self.node_sampler, self.minibatch): bid for bid in range(num_iter_pt)}
                # sampling_jobs = {executor.submit(self.node_sampler, self.minibatch()): bid for bid in range(num_iter) }
                # seeds = self.minibatch()
                # sampling_jobs = {executor.submit(self.node_sampler, seeds): bid for bid in range(num_iter) }

                for future in concurrent.futures.as_completed(sampling_jobs):
                    yield (future.result())
