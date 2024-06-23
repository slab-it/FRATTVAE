import argparse
import os

# set parser
parser = argparse.ArgumentParser(description= 'please enter paths')
parser.add_argument('yml', type= str, help= 'yml file')
parser.add_argument('--load_epoch', type= int, default= None, help= 'load model at load epoch')

# for generation
parser.add_argument('--max_nfrags', type= int, default= None, help= 'max iteration of tree decode')
parser.add_argument('--N', type= int, default= 5, help= 'generate k mols for N times, default N= 5')
parser.add_argument('--k', type= int, default= 10000, help= 'generate k mols, default k= 10000')
parser.add_argument('--t', type= float, default= 1.0, help= 'temperature for generation')

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
import moses
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions.multivariate_normal import MultivariateNormal

from models.fttvae import FTTVAE
from models.property import propLinear
from process import generate
from utils.apps import second2date
from utils.data import collate_pad_fn
from utils.preprocess import SmilesToMorganFingetPrints

if args.recon == args.gen:
    args.recon = args.gen = True

yml_file = args.yml

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
max_nfrags = decomp_params['max_nfrags'] if args.max_nfrags is None else args.max_nfrags
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
print(f'data: {data_path}, useChiral: {useChiral}, n_jobs: {args.n_jobs}', flush= True)
print(f'fragments: {len(uni_fragments)}, feature: {frag_ecfps.shape[-1]}, tree: ({max_depth}, {max_degree}), prop: {prop_dim}', flush= True)

# load model
num_labels = frag_ecfps.shape[0]
if prop_dim:
    pmodel = propLinear(latent_dim, prop_dim).to(device)
    if args.load_epoch:
        pmodel.load_state_dict(torch.load(os.path.join(result_path, 'models', f'pmodel_iter{args.load_epoch}.pth'), map_location= device))
    else:
        pmodel.load_state_dict(torch.load(os.path.join(result_path, 'models', f'pmodel_best.pth'), map_location= device))
    pmodel.eval()
else:
    pmodel = None
model = FTTVAE(num_labels, max_depth, max_degree, feat_dim, latent_dim, 
               d_model, d_ff, num_layers, num_heads, activation).to(device)
if args.load_epoch:
    load_epoch = args.load_epoch
    model.load_state_dict(torch.load(os.path.join(result_path, 'models', f'model_iter{load_epoch}.pth'), map_location= device))
    print(f'model loaded: {os.path.join(result_path, "models", f"model_iter{load_epoch}.pth")}\n', flush= True)
else:
    load_epoch = '_best'
    model.load_state_dict(torch.load(os.path.join(result_path, 'models', f'model_best.pth'), map_location= device))
    print(f'model loaded: {os.path.join(result_path, "models", f"model_best.pth")}\n', flush= True)
model.PE._update_weights()      # initialization
model.eval()

## generation
s = time.time()
print(f'---{datetime.datetime.now()}: Generation start.---', flush= True)

z_mean = torch.zeros(latent_dim)
z_var = torch.ones(latent_dim)
dist = MultivariateNormal(z_mean, args.t * z_var * torch.eye(latent_dim))

for i in range(args.N):
    torch.manual_seed(i)
    z_gen = dist.sample((args.k,))
    dataloader = DataLoader(TensorDataset(z_gen), batch_size= batch_size, shuffle= False)

    z_list, dec_smiles, pred_list, cosines, euclids = generate(dataloader, uni_fragments, frag_ecfps, ndummys, 
                                                               model, pmodel, max_nfrags, useChiral, args.free_n, args.n_jobs, device)                                    
    # eval
    df_gen = pd.DataFrame({'SMILES': dec_smiles, 'cosine': cosines, 'euclid': euclids})
    if pred_list:
        for j, pred in enumerate(zip(*pred_list)):
            df_gen[f'pred{j}'] = pred

    # save results
    with open(os.path.join(result_path, 'generate', f'z_gen_list{load_epoch}_{i}.pkl'), 'wb') as f:
        pickle.dump(z_list, f)
    df_gen.to_csv(os.path.join(result_path, 'generate', f'generate{load_epoch}_{i}.csv'), index= False)

    print(f'[{i+1}/{args.N}] elapsed time: {second2date(time.time()-s)})\n', flush= True)

print(f'---{datetime.datetime.now()}: Generation done. (elapsed time: {second2date(time.time()-s)})---\n', flush= True)