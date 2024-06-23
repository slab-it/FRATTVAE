import argparse
import os

# set parser
parser = argparse.ArgumentParser(description= 'please enter paths')
parser.add_argument('yml', type= str, help= 'yml file')
parser.add_argument('--load_epoch', type= int, default= None, help= 'load model at load epoch')
parser.add_argument('--root', action= 'store_true', help= 'if True, use root node value, else sum all nodes values.')
parser.add_argument('--base', action= 'store_true', help= 'calc attention using model before fine-tuning.')

parser.add_argument('--gpu', type= int, default= 0, help= 'gpu device ids')
parser.add_argument('--n_jobs', type= int, default= 1, help= 'the number of cpu for parallel, default 24')
parser.add_argument('--free_n', action= 'store_true')
args = parser.parse_args()

import datetime
import gc
import pickle
import sys
import time
import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import yaml

import torch
from torch.utils.data import Subset, DataLoader

from models.frattvae import FRATTVAE
from process import featurelize_fragments
from utils.apps import second2date
from utils.data import collate_pad_fn
from utils.preprocess import SmilesToMorganFingetPrints

yml_file = args.yml

start = time.time()
print(f'---{datetime.datetime.now()}: start.---', flush= True)

## check environments
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(args.gpu)
print(f'GPU [{args.gpu}] is available: {torch.cuda.is_available()}\n', flush= True)

## load hyperparameters
with open(yml_file) as yml:
    params = yaml.safe_load(yml)
print(f'load: {yml_file}', flush= True)

# path
result_path= params['result_path']
data_path = params['data_path']
frag_path = params['frag_path']

# hyperparameters for decomposition and tree-fragments
decomp_params = params['decomp']
n_bits = decomp_params['n_bits']
max_nfrags = decomp_params['max_nfrags']
dupl_bits = decomp_params['dupl_bits']
radius = decomp_params['radius']
max_depth = decomp_params['max_depth']
max_degree = decomp_params['max_degree']
useChiral = decomp_params['useChiral']
ignore_double = decomp_params['ignore_double']
ignore_dummy = decomp_params['ignore_dummy']

# hyperparameters for model
model_params = params['model']
d_model = model_params['d_model']
d_ff = model_params['d_ff']
num_layers = model_params['nlayer']
num_heads = model_params['nhead']
activation = model_params['activation']
latent_dim = model_params['latent']
feat_dim = model_params['feat']
props = model_params['property']
pnames = list(props.keys())
ploss = model_params['ploss']

# hyperparameters for training
train_params = params['train']
batch_size = train_params['batch_size']
# batch_size = 128

## load data
df = pd.read_csv(data_path)
df_frag = pd.read_csv(frag_path)
uni_fragments = df_frag['SMILES'].tolist()
freq_list = df_frag['frequency'].tolist()
try:
    with open(os.path.join(result_path, 'input_data', 'csr_ecfps.pkl'), 'rb') as f:
        frag_ecfps = pickle.load(f).toarray()
        frag_ecfps = torch.from_numpy(frag_ecfps).float()
    assert frag_ecfps.shape[0] == len(uni_fragments)
    assert frag_ecfps.shape[1] == (n_bits + dupl_bits)
except Exception as e:
    print(e, flush= True)
    frag_ecfps = torch.tensor(SmilesToMorganFingetPrints(uni_fragments[1:], n_bits= n_bits, dupl_bits= dupl_bits, radius= radius, 
                                                         ignore_dummy= ignore_dummy, useChiral= useChiral, n_jobs= args.n_jobs)).float()
    frag_ecfps = torch.vstack([frag_ecfps.new_zeros(1, n_bits+dupl_bits), frag_ecfps])      # padding feature is zero vector
ndummys = torch.tensor(df_frag['ndummys'].tolist()).long()
prop_dim = sum(list(props.values())) if pnames else None
print(f'data: {data_path}', flush= True)
print(f'train: {sum(df.test==0)}, valid: {sum(df.test==-1)}, test: {sum(df.test==1)}, useChiral: {useChiral}, n_jobs: {args.n_jobs}', flush= True)
print(f'fragments: {len(uni_fragments)}, feature: {frag_ecfps.shape[-1]}, tree: ({max_depth}, {max_degree}), prop: {prop_dim}', flush= True)

# load model
num_labels = frag_ecfps.shape[0]
model = FRATTVAE(num_labels, max_depth, max_degree, feat_dim, latent_dim, 
               d_model, d_ff, num_layers, num_heads, activation).to(device)
if args.base:
    load_epoch = '_base'
    model.load_state_dict(torch.load(params['model_path'], map_location= device))
    print(f'model loaded: {params["model_path"]}\n', flush= True)
elif args.load_epoch:
    load_epoch = args.load_epoch
    model.load_state_dict(torch.load(os.path.join(result_path, 'models', f'model_iter{load_epoch}.pth'), map_location= device))
    print(f'model loaded: {os.path.join(result_path, "models", f"model_iter{load_epoch}.pth")}\n', flush= True)
else:
    load_epoch = '_best'
    model.load_state_dict(torch.load(os.path.join(result_path, 'models', f'model_best.pth'), map_location= device))
    print(f'model loaded: {os.path.join(result_path, "models", f"model_best.pth")}\n', flush= True)
model.PE._update_weights()      # initialization
model.eval()

# get attention between root and fragments
s = time.time()
print(f'---{datetime.datetime.now()}: Getting Attention start.---', flush= True)
with open(os.path.join(result_path, 'input_data', 'dataset.pkl'), 'rb') as f:
    dataset = pickle.load(f)

dataset = Subset(dataset, df.loc[df.test==0].index.tolist())
dataloader = DataLoader(dataset, batch_size= batch_size, shuffle= False, pin_memory= True, 
                        num_workers= 4, collate_fn= collate_pad_fn)
attn_mean, attn_std = featurelize_fragments(dataloader, model, frag_ecfps, args.root, device)

mean_dict = {f'layer{i}': layer for i, layer in enumerate(zip(*attn_mean))}
std_dict = {f'std{i}': layer for i, layer in enumerate(zip(*attn_std))}
avg_mean = np.mean(attn_mean, axis= 1).squeeze()
avg_std = np.mean(attn_std, axis= 1).squeeze()
df_attn = pd.DataFrame({'avg': avg_mean, 'avg_std': avg_std} | mean_dict | std_dict)
tmp = 'root' if args.root else 'mean'
df_attn.to_csv(os.path.join(result_path, 'visualize', f'attention{load_epoch}_{tmp}.csv'), index= False)

print('featurelized fragments:', flush= True)
df_attn = df_attn.sort_values(by= 'avg', ascending= False)
for i in range(5):
    idx = df_attn.index[i]
    print(f'- {uni_fragments[idx]}: {df_attn["avg"].iloc[i]:.4f}, (std; {df_attn["avg_std"].iloc[i]:.4f})', flush= True)

print(f'---{datetime.datetime.now()}: Getting Attention done. (elapsed time: {second2date(time.time()-s)})---', flush= True)

