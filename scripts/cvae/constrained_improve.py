import argparse
import os

# set parser
parser = argparse.ArgumentParser(description= 'please enter paths')
parser.add_argument('yml', type= str, help= 'yml file')
parser.add_argument('mols', type= str, default= None, help= 'a file of SMILES that are the starting point of generation. SMILES txt or csv which has SMILES column')
parser.add_argument('key', type= str, help= 'condition key for improvement. ex: MW, QED, SA, logP, NP etc.')
parser.add_argument('target', type= float, help= 'condition max value for improvement. Add value to origin value.')
parser.add_argument('--k', type= int, default= 80, help= 'generate k mols / mol')
parser.add_argument('--thresholds', type= float, default= [0.2, 0.4, 0.6], nargs='*', help= 'list of similarity cutoff, default= [0.2, 0.4, 0.6]')
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
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem
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

if args.key not in pnames:
    raise KeyError(f'{args.key} is not in conditions')

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
print(f'key-target: {args.key} {args.target}\n', flush= True)
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
df_props['delta'] = (args.target - df_props[args.key]) / args.k
originals[args.key] = df_props[args.key].tolist()

smiles_opt = []
for _ in range(args.k):
    df_props[args.key] += df_props['delta']
    conditions = torch.from_numpy(np.array([normalize(df_props[p].to_numpy(), p) for p in pnames]).T).float()

    # z to smiles
    dataloader = DataLoader(TensorDataset(z_all, conditions), batch_size= batch_size, shuffle= False)
    _, dec_smiles = generate(dataloader, uni_fragments, frag_ecfps, ndummys, model, pnames,
                             max_nfrags, useChiral, args.free_n, args.n_jobs, device)
    smiles_opt.append(dec_smiles)

scoring_func = METRICS_DICT[args.key]
improved_smiles, improved_score = {f'thr-{thr}': [] for thr in args.thresholds}, {f'thr-{thr}': [] for thr in args.thresholds}
for mol, smis_opt in zip(mols, zip(*smiles_opt)):
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
    ecfps_opt = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048) for s in smis_opt]
    tanimotos = DataStructs.BulkTanimotoSimilarity(ecfp, ecfps_opt)
    tanimotos = np.where(tanimotos==1, 0, tanimotos)    # remove the same mols

    orig_score = scoring_func(mol)
    for thr in args.thresholds:
        smis_cand = [smis_opt[i] for i in range(args.k) if tanimotos[i] >= thr]
        scores = np.array([scoring_func(Chem.MolFromSmiles(s)) for s in smis_cand])

        if np.any(scores>orig_score):
            max_improve = scores.max() - orig_score
            improved_smi = smis_cand[np.argmax(scores)]
        else:
            max_improve = float('nan')
            improved_smi = None
        improved_smiles[f'thr-{thr}'].append(improved_smi)
        improved_score[f'thr-{thr}'].append(max_improve)

print(f'Threshold, Improved score, Success rate (elapsed time: {second2date(time.time()-s)})', flush= True)
for thr in args.thresholds:
    key = f'thr-{thr}'
    total = len(improved_smiles[key])
    print(f'- {thr}: {np.nanmean(improved_score[key]):.4f} (std; {np.nanstd(improved_score[key]):.4f}), {(total-improved_smiles[key].count(None))/total:.4f}', flush= True)
df_gen = pd.DataFrame({'SMILES': smiles} | improved_smiles)
df_gen.to_csv(os.path.join(result_path, 'generate', f'const_improvement_{load_epoch}.csv'), index= False)

print(f'---{datetime.datetime.now()}: all process done. (elapsed time: {second2date(time.time()-start)})---', flush= True)