import argparse
import os

# set parser
parser = argparse.ArgumentParser(description= 'please enter paths')
parser.add_argument('yml', type= str, help= 'yml file')
parser.add_argument('--keys', type= str, default= [], nargs='*', help= 'a list of condition keys for generation. ex: MW, QED, SA, logP, NP etc.')
parser.add_argument('--values', type= float, default= [], nargs='*', help= 'a list of condition values for generation.')
parser.add_argument('--load_epoch', type= int, default= None, help= 'load model at load epoch')

parser.add_argument('--N', type= int, default= 1, help= 'generate k mols for N times, default N= 1')
parser.add_argument('--k', type= int, default= 10000, help= 'generate k mols, default k= 10000')
parser.add_argument('--t', type= float, default= 1.0, help= 'temperature for z-generation')

parser.add_argument('--gpu', type= int, default= 0, help= 'gpu device ids')
parser.add_argument('--n_jobs', type= int, default= 1, help= 'the number of cpu for parallel, default 24')
parser.add_argument('--free_n', action= 'store_true')
args = parser.parse_args()

import datetime
import pickle
import sys
import time
import warnings
from joblib import Parallel, delayed
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import yaml
from rdkit import RDLogger
from moses.metrics import metrics
RDLogger.DisableLog('rdApp.*')
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

import torch
import torch.distributions as dist
from torch.utils.data import DataLoader, TensorDataset

from models.frattvae import FRATTVAE
from models.wrapper import CVAEwrapper
from cvae.process import generate
from utils.apps import second2date
from utils.preprocess import SmilesToMorganFingetPrints
from utils.chem_metrics import normalize, get_all_metrics, METRICS_DICT

if len(args.keys) == 0:
    raise ValueError('please input condition keys.')
assert len(args.keys) == len(args.values)

start = time.time()
print(f'---{datetime.datetime.now()}: start.---', flush= True)

## check environments
if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.set_device(args.gpu)
else:
    device = 'cpu'
print(f'GPU [{args.gpu}] is available: {torch.cuda.is_available()}\n', flush= True)

## load hyperparameters
yml_file = args.yml
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
min_size = decomp_params['min_size']
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

# hyperparameters for training
train_params = params['train']
batch_size = train_params['batch_size']
# batch_size = 128

## load data
df = pd.read_csv(data_path)
df_frag = pd.read_csv(os.path.join(result_path, 'input_data', 'fragments.csv'))
uni_fragments = df_frag['SMILES'].tolist()
freq_list = df_frag['frequency'].tolist()
cond_dim = len(pnames) if pnames else None

for key in args.keys:
    if key not in pnames:
        raise KeyError(f'{key} is not in conditions')

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

# load model
num_labels = frag_ecfps.shape[0]
model = FRATTVAE(num_labels, max_depth, max_degree, feat_dim, latent_dim,
               d_model, d_ff, num_layers, num_heads, activation).to(device)
model = CVAEwrapper(model, pnames, list(props.values())).to(device)
if args.load_epoch:
    load_epoch = args.load_epoch
    model.load_state_dict(torch.load(os.path.join(result_path, 'models', f'model_iter{load_epoch}.pth'), map_location= device))
    print(f'model loaded: {os.path.join(result_path, "models", f"model_iter{load_epoch}.pth")}\n', flush= True)
else:
    load_epoch = '_best'
    model.load_state_dict(torch.load(os.path.join(result_path, 'models', f'model_best.pth'), map_location= device))
    print(f'model loaded: {os.path.join(result_path, "models", f"model_best.pth")}\n', flush= True)
model.vae.PE._update_weights()      # initialization
model.eval()

print(f'generate: {args.k}', flush= True)
print(f'fragments: {len(uni_fragments)}, feature: {frag_ecfps.shape[-1]}, tree: ({max_depth}, {max_degree}), cond: {cond_dim}', flush= True)
print(f'targets:', ', '.join([f'{k} {v}' for k, v in zip(args.keys, args.values)]), '\n', flush= True)
print('[ORIGINAL]', ", ".join([f"{key}: {np.nanmean(df[key]):.4f}" for key in args.keys]), flush= True)

# define distributions
multinorm = dist.multivariate_normal.MultivariateNormal(torch.zeros(latent_dim), args.t * torch.ones(latent_dim) * torch.eye(latent_dim))

METRICS, METRICS_TEST = {}, {}
for i in range(args.N):
    s = time.time()
    z_all = multinorm.sample((args.k,))
    conditions = torch.hstack([torch.from_numpy(normalize(np.array([val for _ in range(args.k)]), key)).view(-1, 1).float() for key, val in zip(args.keys, args.values)])

    # z to smiles
    dataloader = DataLoader(TensorDataset(z_all, conditions), batch_size= batch_size, shuffle= False)
    z_list, dec_smiles = generate(dataloader, uni_fragments, frag_ecfps, ndummys, model, args.keys,
                                  max_nfrags, useChiral, args.free_n, args.n_jobs, device)

    # eval
    properties = Parallel(n_jobs= args.n_jobs)(delayed(get_all_metrics)(s) for s in dec_smiles)
    prop_dict = {f'{key}': list(prop) for key, prop in zip(METRICS_DICT.keys(), zip(*properties))}
    df_gen = pd.DataFrame({'SMILES': dec_smiles} | prop_dict)

    # moses
    dec_smiles = metrics.remove_invalid(dec_smiles, n_jobs= args.n_jobs)
    novels = metrics.novelty(dec_smiles, df.SMILES.tolist(), n_jobs= args.n_jobs)
    uniques = metrics.fraction_unique(dec_smiles, n_jobs= args.n_jobs)

    # save results
    # with open(os.path.join(result_path, 'generate', f'z_gen_list{load_epoch}_{"_".join([f"{k}{v}" for k, v in zip(args.keys, args.values)])}_{i}.pkl'), 'wb') as f:
    #     pickle.dump(z_list, f)
    df_gen.to_csv(os.path.join(result_path, 'generate', f'generate{load_epoch}_{"_".join([f"{k}{v}" for k, v in zip(args.keys, args.values)])}_{i}.csv'), index= False)
    print(f'[{i+1}/{args.N}] valid: {len(dec_smiles)/args.k:.4f}, unique: {uniques:.4f}, novelty: {novels:.4f} (elapsed time: {second2date(time.time()-s)})')
    for n, prop in prop_dict.items():
        print(f'- {n}: {np.nanmean(prop):.4f} (std; {np.nanstd(prop):.4f}, range; {np.nanmin(prop):.4f}-{np.nanmax(prop):.4f})', flush= True)

print(f'\n---{datetime.datetime.now()}: all process done. (elapsed time: {second2date(time.time()-start)})---', flush= True)