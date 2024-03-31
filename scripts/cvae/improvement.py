import argparse
import os

# set parser
parser = argparse.ArgumentParser(description= 'please enter paths')
parser.add_argument('yml', type= str, help= 'yml file')
parser.add_argument('mols', type= str, default= None, help= 'a file of SMILES that are the starting point of generation. SMILES txt or csv which has SMILES column')
parser.add_argument('--keys', type= str, default= [], nargs='*', help= 'a list of condition keys for generation. ex: MW, QED, SA, logP, NP etc.')
parser.add_argument('--values', type= float, default= [], nargs='*', help= 'a list of condition values for generation. In improvement, add value to origin value.')
parser.add_argument('--load_epoch', type= int, default= None, help= 'load model at load epoch')

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
from joblib import Parallel, delayed
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset

from models.fttvae import FTTVAE
from models.wrapper import CVAEwrapper
from cvae.process import generate
from utils.apps import second2date
from utils.mask import create_mask
from utils.tree import get_tree_features
from utils.preprocess import smiles2mol, SmilesToMorganFingetPrints, parallelMolsToBRICSfragments
from utils.construct import calc_tanimoto
from utils.chem_metrics import get_all_metrics, normalize, METRICS_DICT


if len(args.keys) == 0:
    raise ValueError('please input condition keys.')
assert len(args.keys) == len(args.values)

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

# load mols
if '.csv' in args.mols:
    df_mols = pd.read_csv(args.mols)
    smiles = df_mols.SMILES.tolist()
    del df_mols
elif '.txt' in args.mols:
    with open(args.mols) as f:
        smiles = [line.rstrip('\n') for line in f.readlines()]
else:
    raise ValueError('only accept txt or csv flie.')

mols = Parallel(n_jobs= args.n_jobs)(delayed(smiles2mol)(s) for s in smiles)
properties = Parallel(n_jobs= args.n_jobs)(delayed(get_all_metrics)(m) for m in mols)
df_props = pd.DataFrame({f'{key}': list(values) for key, values in zip(METRICS_DICT.keys(), zip(*properties))})

fragments_list, bondtypes_list, bondMapNums_list \
, recon_flag, uni_fragments, freq_label = parallelMolsToBRICSfragments(mols,
                                                                       minFragSize = min_size, maxFragNums= max_nfrags, maxDegree= max_degree,
                                                                       useChiral= useChiral, ignore_double= ignore_double, 
                                                                       df_frag= df_frag, asFragments= False,
                                                                       n_jobs= args.n_jobs, verbose= 0)
df_frags = pd.DataFrame({'SMILES': uni_fragments, 'frequency': freq_label, 'ndummys': [f.count('*') for f in uni_fragments]})
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
num_labels = len(df_frag)
model = FTTVAE(num_labels, max_depth, max_degree, feat_dim, latent_dim,
               d_model, d_ff, num_layers, num_heads, activation)
model = CVAEwrapper(model, pnames, list(props.values())).to(device)
if args.load_epoch:
    load_epoch = args.load_epoch
    model.load_state_dict(torch.load(os.path.join(result_path, 'models', f'model_iter{load_epoch}.pth'), map_location= device))
    print(f'model loaded: {os.path.join(result_path, "models", f"model_iter{load_epoch}.pth")}\n', flush= True)
else:
    load_epoch = 'best'
    model.load_state_dict(torch.load(os.path.join(result_path, 'models', f'model_best.pth'), map_location= device))
    print(f'model loaded: {os.path.join(result_path, "models", f"model_best.pth")}\n', flush= True)
model.eval()

print(f'data: {args.mols}, generate: {len(mols)}', flush= True)
print(f'fragments: {len(uni_fragments)}, feature: {frag_ecfps.shape[-1]}, tree: ({max_depth}, {max_degree}), cond: {cond_dim}', flush= True)
print(f'key-values:', ', '.join([f'{k} {v}' for k, v in zip(args.keys, args.values)]), '\n', flush= True)
print(f'[BEFORE]', ", ".join([f"{key}: {np.nanmean(values):.4f}" for key, values in df_props.items()]), flush= True)


## improvement
s = time.time()

# mols to trees
trees = Parallel(n_jobs= args.n_jobs)(delayed(get_tree_features)(f, frag_ecfps[f], b, m, max_depth, max_degree, args.free_n) for f, b, m in zip(fragments_list, bondtypes_list, bondMapNums_list))
frag_indices, features, positions = zip(*trees)
frag_indices = pad_sequence(frag_indices, batch_first= True, padding_value= 0)
features = pad_sequence(features, batch_first= True, padding_value= 0)
positions = pad_sequence(positions, batch_first= True, padding_value= 0)
conditions = torch.from_numpy(np.array([normalize(df_props[p].to_numpy(), p) for p in pnames]).T).float()   # normalized
nan_mask = torch.where(conditions.isnan(), 0, 1)
src = torch.hstack([nan_mask, torch.full((frag_indices.shape[0], 1), -1), frag_indices.detach()])  # for super root & conditions
src_mask, _, src_pad_mask, _ = create_mask(src, src, pad_idx= 0, batch_first= True)

dataloader = DataLoader(TensorDataset(features, positions, conditions, src_pad_mask), batch_size= batch_size, shuffle= False)

# mols to latent variables
z_all = []
with torch.no_grad():
    for data in dataloader:
        f, p, c, pm = data
        conditions = {key: c[:, i].to(device) for i, key in enumerate(pnames)}
        z, _, _ = model.encode(f.to(device), p.to(device), conditions, src_mask.to(device), pm.to(device))
        z_all.append(z.cpu())
z_all = torch.vstack(z_all)

# add values to conditions
originals = {}
for k, v in zip(args.keys, args.values):
    originals[k] = df_props[k].tolist()
    df_props[k] += v
conditions = torch.from_numpy(np.array([normalize(df_props[p].to_numpy(), p) for p in pnames]).T).float()

# z to smiles
dataloader = DataLoader(TensorDataset(z_all, conditions), batch_size= batch_size, shuffle= False)
z_list, dec_smiles = generate(dataloader, uni_fragments, frag_ecfps, ndummys, model, pnames,
                              max_nfrags, useChiral, args.free_n, args.n_jobs, device)

# eval
tanimotos = Parallel(n_jobs= args.n_jobs)(delayed(calc_tanimoto)(smi1, smi2, useChiral= useChiral) for smi1, smi2 in zip(smiles, dec_smiles))
print(f'tanimoto simirality: <mean> {np.nanmean(tanimotos):.6f}, <std> {np.nanstd(tanimotos):.6f}\n', flush= True)

properties = Parallel(n_jobs= args.n_jobs)(delayed(get_all_metrics)(s) for s in dec_smiles)
prop_dict = {f'{key}': list(prop) for key, prop in zip(METRICS_DICT.keys(), zip(*properties))}
df_gen = pd.DataFrame({'SMILES': smiles, 'IMPROVE': dec_smiles, 'tanimoto': tanimotos} | prop_dict)

# save results
fname = args.mols.split('/')[-1].split('.')[0]
df_gen.to_csv(os.path.join(result_path, 'generate', f'{fname}_{"_".join([f"{k}{v}" for k, v in zip(args.keys, args.values)])}_{load_epoch}.csv'), index= False)
print('[AFTER]', ", ".join([f"{key}: {np.nanmean(df_gen[key]):.4f}" for key in METRICS_DICT.keys()]), f'elapsed time: {second2date(time.time()-s)})', flush= True)

print(f'[IMPROVE]', flush= True)
success_flag = np.full(shape= (len(mols),), fill_value= True)
for k in args.keys:
    orig = np.array(originals[k])
    genr = df_gen[k].to_numpy()
    impr = genr - orig
    success = (impr > 0).sum()
    success_flag = success_flag & (impr > 0)
    if success > 0:
        impr = impr[impr>0]
    print(f'- {k}: {np.mean(genr):.4f} (std: {np.std(genr):.4f}) / success-rate {success/len(smiles):.4f} ({success}/{len(smiles)}), improve {np.mean(impr):.4f} (std: {np.std(impr):.4f})', flush= True)
print(f'all conditions improved: {success_flag.sum()/len(smiles):.6f} ({success_flag.sum()}/{len(smiles)})\n', flush= True)

print(f'---{datetime.datetime.now()}: all process done. (elapsed time: {second2date(time.time()-start)})---', flush= True)