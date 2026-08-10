# coding: utf-8
from pathlib import Path
import torch

from alpha_analysis.ai.dataloader import Ascot5Dataset
from alpha_analysis.ai.train_transolver import (
    _discover_sample_folders,
    _split_indices,
    sample_to_transolver_tensors,
)

results_root = Path("/global/cfs/cdirs/m5300/results/G1600")

folders = _discover_sample_folders(
    results_root,
    "analysis_results.h5",
    "desc_equilibrium.h5",
    "bfield.h5",
)

dataset = Ascot5Dataset(
    folders,
    analysis_filename="analysis_results.h5",
    equilibrium_filename="desc_equilibrium.h5",
    bfield_filename="bfield.h5",
    include_bfield=True,
    strict=True,
)

train_indices, val_indices = _split_indices(len(dataset), train_fraction=0.8, seed=0)

len(dataset), len(train_indices), len(val_indices)
sample = dataset[292]
sample.keys()
g = torch.Generator().manual_seed(0)

x, pos, y = sample_to_transolver_tensors(
    sample,
    max_nodes=16384,
    target_reduction="mean",
    log10_target=True,
    target_eps=1e-30,
    profile_log1p=True,
    generator=g,
)

x.shape, pos.shape, y
x
pos
get_ipython().run_line_magic('pwd', '')
get_ipython().run_line_magic('ls', '')
get_ipython().run_line_magic('ls', 'alpha_analysis/')
get_ipython().run_line_magic('ls', 'alpha_analysis/ai')
get_ipython().system('code alpha_analysis/ai/train_transolver.py')
sample
sample.keys()
sample['bfield'].keys()
import matplotlib.pyplot as plt; plt.ion()
sample['prs_para'].shape
sample['bfield']['rho'].shape
sample['bfield']['phi'].shape
sample['bfield']['theta'].shape
plt.plot(np.flatten(sample['bfield']['rho'][None,:,None,None]),np.flatten(sample['prs_para'][0,...]),'.')
import numpy as np
plt.plot(np.flatten(sample['bfield']['rho'][None,:,None,None]),np.flatten(sample['prs_para'][0,...]),'.')
plt.plot(sample['bfield']['rho'][None,:,None,None],sample['prs_para'][0,...],'.')
plt.plot(sample['bfield']['rho'][:,None,None],sample['prs_para'][0,...],'.')
plt.plot(sample['bfield']['rho'][:,None,None].expand_as(sample['prs_para'][0,...]),sample['prs_para'][0,...],'.')
rho = sample['bfield']['rho']
prs = sample['prs_para'][0]
rho3 = rho[:, None, None].expand_as(prs)
plt.plot(rho3.flatten(), prs.flatten(), ".", markersize=1)
plt.xlabel("rho")
plt.ylabel("prs_para[0]")
get_ipython().run_line_magic('pwd', '')
plt.savefig('tmp.png')
get_ipython().system('code tmp.png')
prs = sample['prs_para'][4]
plt.plot(rho3.flatten(), prs.flatten(), ".", markersize=1)
plt.xlabel("rho")
plt.ylabel("prs_para[0]")
prs = sample['prs_para'][-1]
plt.plot(rho3.flatten(), prs.flatten(), ".", markersize=1)
plt.xlabel("rho")
plt.ylabel("prs_para[0]")
plt.savefig('tmp.png')
sample = dataset[100]
rho = sample['bfield']['rho']
prs = sample['prs_para'][0]
rho3 = rho[:, None, None].expand_as(prs)
plt.clf()
plt.plot(rho3.flatten(), prs.flatten(), ".", markersize=1)
plt.xlabel("rho")
plt.ylabel("prs_para[0]")
prs = sample['prs_para'][4]
plt.plot(rho3.flatten(), prs.flatten(), ".", markersize=1)
plt.xlabel("rho")
plt.ylabel("prs_para[0]")
prs = sample['prs_para'][-1]
plt.plot(rho3.flatten(), prs.flatten(), ".", markersize=1)
plt.xlabel("rho")
plt.ylabel("prs_para[0]")
plt.savefig('tmp.png')
dataset[292]['folder']
sample.keys()
sample['target']
sample['target'].min()
sample['target'].max()
sample['target'].mean()
sample['target'].shape
plt.clf()
plt.hist(sample['target'],1000)
plt.savefig('tmp.png')
sample = dataset[292]
plt.hist(sample['target'],1000)
plt.savefig('tmp.png')
plt.clf()
plt.hist(sample['target'],1000)
plt.savefig('tmp.png')
sample['target'].max()
sample_good = dataset[100]
sample_good['target'].min()
sample_good['target'].max()
sample['target'].min()
sample['target'].max()
sample_good['target'].min(), sample_good['target'].max()
sample['target'].min(), sample['target'].max()
np.sum(sample['target']>1)
np.sum(sample['target']>1.0)
tmp = sample['target']>1.0
tmp
np.array(tmp)
tmp.dtype
np.sum((sample['target']>1.0).long())
tmp = sample['target']>1.0
tmp.numpy()
sample['target'].shape
np.sum(tmp.numpy())
plt.clf()
plt.hist(sample_good['target'],1000)
plt.savefig('tmp.png')
tmp[tmp>1] = 0.0
tmp.max()
tmp = sample['target']
tmp[tmp>1] = 0.0
tmp.max)
tmp.max()
plt.hist(tmp,1000)
plt.savefig('tmp.png')
get_ipython().run_line_magic('pwd', '')
get_ipython().run_line_magic('ls', '')
get_ipython().run_line_magic('save', '-f log_sample_explore.py 1-1000')
