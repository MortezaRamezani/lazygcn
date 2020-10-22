import os
import sys
import torch
import numpy as np

import lazygcn
from lazygcn import data, training, models, utils, samplers, profiler

# set the dataset folder
os.environ["GNN_DATASET_DIR"] = "/export/local/mfr5226/datasets/"

# number of repeatation
rep = int(sys.argv[1])

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

args = utils.RunConfig('Comparing RegularGCN with LazyGCN', [
                       '--prefix', './results/'])

args.update_args({'gpu': 0,
                  'num_workers': 8,
                  'dropout': 0,
                  'lr': 1e-3,
                  'train_batch_size':  2048*2,
                  'hidden_dim': 512,
                  'num_layers': 2,
                  'minibatch_method': 'random',
                  'supervised': True,
                  'log': True,
                  'no_val': False,
                  'val_frequency': 1,
                  'rec_subsample': 'outer',
                  'concat': True,
                  'num_iters': 1,
                  'num_epochs': 1,
                  })

all_trainers = []

# Pubmed
# dataset = data.Dataset('pubmed')
# print(dataset)
# model = models.RecycleGCN
# args.update_args({'train_batch_size':  2048*2,'num_epochs': 200,})
# all_mode = {
#     'ladies': [1, 1, 'ladies', 512],
#     'ladies+lg': [1.1, 2, 'ladies', 512],
# }


#Reddit
dataset = data.Dataset('reddit')
print(dataset)
model = models.RecycleGCN
args.update_args({'train_batch_size':  2048*8,'num_epochs': 200, 'val_frequency': 10, })
all_mode = {
    'ladies': [1, 1, 'ladies', 512],
    'ladies+lg': [1.1, 2, 'ladies', 512],
}


all_results = {}

for mode in all_mode.keys():

    rho = all_mode[mode][0]
    rp = all_mode[mode][1]
    sm = all_mode[mode][2]
    ss = all_mode[mode][3]
    print(mode, rho, rp, sm, ss)

    all5_time = []
    all5_score = []

    all_results[mode] = []

    for i in range(rep):
        postfix = '{}({})'.format(mode, i)
        args.update_args({'rho': rho,
                          'recycle_period': rp,
                          'sampling_method': sm,
                          'train_sample_size': ss,
                          'postfix': postfix,
                          })

        trainer = training.RecycleExpGCN(
            args.config, dataset, model, samplers.Sampler)

        trainer.run()
        trainer.save()

        all5_score.append(trainer.test_score*100)
        all5_time.append(trainer.train_time)

        all_results[mode].append((trainer.test_score, trainer.train_time))

    print('>>'*100)
    print('>>RESULT for {}, score {:.2f}±{:.2f}, time {:.2f}±{:.2f} (s)'.format(
          mode,
          np.mean(all5_score), np.var(all5_score),
          np.mean(all5_time), np.var(all5_time)
          ))
    print('<<'*100)
        

np.save(trainer.config.log_dir+'/all-res.npy', all_results)
