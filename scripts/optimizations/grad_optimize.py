import argparse
import os

# set parser
parser = argparse.ArgumentParser(description= 'please enter paths')
parser.add_argument('yml', type= str, help= 'yml file')
parser.add_argument('--load_epoch', type= int, default= None, help= 'load model at load epoch')

# for constrained optimize
parser.add_argument('--constrained', type= str, default= None, help= 'smiles txt file for constrained property optimization')
parser.add_argument('--thresholds', type= float, default= [0.2, 0.4, 0.6], nargs='*', help= 'list of similarity cutoff, default= [0.2, 0.4, 0.6]')

# for property optimized random generation
parser.add_argument('--k', type= int, default= 10000, help= 'generate k mols, default k= 10000')
parser.add_argument('--seed', type= int, default= 0, help= 'random seed')

parser.add_argument('--target', type= float, default= [1.0], nargs='*', help= 'target values, default= [1.0]')
parser.add_argument('--lr', type= float, default= 0.1, help= 'learning rate for updates, default= 0.1')
parser.add_argument('--max_iter', type= int, default= 80, help= 'max iterations of updates. default= 80')
parser.add_argument('--patient', type= int, default= 5, help= 'max patient. default= 5')
parser.add_argument('--gpu', type= int, default= 0, help= 'gpu device ids')
parser.add_argument('--n_jobs', type= int, default= 1, help= 'the number of cpu for parallel, default 24')
parser.add_argument('--free_n', action= 'store_true')
args = parser.parse_args()

import datetime
import pickle
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
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions.multivariate_normal import MultivariateNormal

from models.frattvae import FRATTVAE
from models.property import propLinear
from process import generate, prop_optimize, constrained_prop_optimize
from utils.apps import second2date
from utils.metrics import CRITERION
from utils.preprocess import SmilesToMorganFingetPrints
from utils.chem_metrics import METRICS_DICT, get_all_metrics

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
min_size = decomp_params['min_size']
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
modes = {'train': 0, 'valid': -1, 'test': 1}
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
ndummys = torch.tensor(df_frag['ndummys'].tolist()).long()
prop_dim = sum(list(props.values())) if pnames else None
print(f'data: {data_path}', flush= True)
print(f'train: {sum(df.test==0)}, valid: {sum(df.test==-1)}, test: {sum(df.test==1)}, useChiral: {useChiral}, n_jobs: {args.n_jobs}', flush= True)
print(f'fragments: {len(uni_fragments)}, feature: {frag_ecfps.shape[-1]}, tree: ({max_depth}, {max_degree}), prop: {prop_dim}', flush= True)

# load model
num_labels = frag_ecfps.shape[0]
pmodel = propLinear(latent_dim, prop_dim).to(device)
if args.load_epoch:
    pmodel.load_state_dict(torch.load(os.path.join(result_path, 'models', f'pmodel_iter{args.load_epoch}.pth'), map_location= device))
else:
    pmodel.load_state_dict(torch.load(os.path.join(result_path, 'models', f'pmodel_best.pth'), map_location= device))
pmodel.eval()
model = FRATTVAE(num_labels, max_depth, max_degree, feat_dim, latent_dim, 
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
criterion = CRITERION[ploss](reduction= 'none')


# property optimize
s = time.time()
if args.constrained:
    print(f'---{datetime.datetime.now()}: Constrained property optimization start.---', flush= True)
    print(f'load: {args.constrained}, thresholds: {args.thresholds}', flush= True)
    print(f'lr: {args.lr}, max_iter: {args.max_iter}, target: {args.target}', flush= True)
    assert len(args.target) == prop_dim == 1
    with open(args.constrained, 'r') as f:
        smiles = [line.rstrip('\n') for line in f.readlines()]
    mols = [Chem.MolFromSmiles(s) for s in smiles]

    improved_smiles, improved_score = {f'thr-{thr}': [] for thr in args.thresholds}, {f'thr-{thr}': [] for thr in args.thresholds}
    scoring_func = METRICS_DICT[pnames[0]]
    for iter, idx in enumerate(range(0, len(mols), batch_size)):
        batched_mols = mols[idx:idx+batch_size]
        target = torch.full(size= (len(batched_mols), 1), fill_value= args.target[0]).float()
        imp_smis, imp_score = constrained_prop_optimize(batched_mols, target, scoring_func, model, pmodel,
                                                        uni_fragments, frag_ecfps, ndummys, 
                                                        min_size, max_nfrags, useChiral, ignore_double,
                                                        criterion, args.thresholds, args.lr, args.max_iter, device)
        for key in improved_smiles.keys():
            improved_smiles[key] += imp_smis[key]
            improved_score[key] += imp_score[key]

        print(f'[{iter+1}/{(len(mols)//batch_size)+1}] elapsed time: {second2date(time.time()-s)}', flush= True)

    print(f'Threshold, Improved score, Success rate (elapsed time: {second2date(time.time()-s)})', flush= True)
    for thr in args.thresholds:
        key = f'thr-{thr}'
        total = len(improved_smiles[key])
        print(f'- {thr}: {np.nanmean(improved_score[key]):.4f} (std; {np.nanstd(improved_score[key]):.4f}), {(total-improved_smiles[key].count(None))/total:.4f}', flush= True)
    df_gen = pd.DataFrame({'SMILES': smiles} | improved_smiles)
    df_gen.to_csv(os.path.join(result_path, 'generate', f'const_opt_generate{load_epoch}.csv'), index= False)

else:
    print(f'---{datetime.datetime.now()}: Property optimization start.---', flush= True)
    print(f'generate: {args.k}, seed: {args.seed}', flush= True)
    print(f'lr: {args.lr}, max_iter: {args.max_iter}, target: {args.target}', flush= True)
    z_mean = torch.zeros(latent_dim)
    z_var = torch.ones(latent_dim)
    dist = MultivariateNormal(z_mean, z_var * torch.eye(latent_dim))

    METRICS, METRICS_TEST = {}, {}
    torch.manual_seed(args.seed)
    z_gen = dist.sample((args.k,))

    z_opt = []
    for iter, idx in enumerate(range(0, args.k, batch_size)):
        z = z_gen[idx:idx+batch_size].view(-1, z_gen.shape[-1])
        if len(args.target) == 1:
            target = torch.full(size= (z.shape[0], prop_dim), fill_value= args.target[0]).float()
        elif len(args.target) == prop_dim:
            target = torch.tensor(args.target).unsqueeze(0).expand(z.shape[0], -1).float()
        else:
            raise ValueError('prop_dim and target.shape[-1] is not equal.')
        z_opt.append(prop_optimize(z, target, pmodel, criterion, args.lr, args.max_iter, args.patient, device))
    z_opt = torch.vstack(z_opt)

    dataloader = DataLoader(TensorDataset(z_opt), batch_size= batch_size, shuffle= False)
    z_list, dec_smiles, pred_list = generate(dataloader, uni_fragments, frag_ecfps, ndummys, 
                                             model, pmodel, max_nfrags, useChiral, args.free_n, args.n_jobs, device)
    properties = Parallel(n_jobs= args.n_jobs)(delayed(get_all_metrics)(s) for s in dec_smiles)
    prop_dict = {f'{key}': list(prop) for key, prop in zip(METRICS_DICT.keys(), zip(*properties))}
    df_gen = pd.DataFrame({'SMILES': dec_smiles} | prop_dict)
    if pred_list:
        for i, pred in enumerate(zip(*pred_list)):
            df_gen[f'pred-{pnames[i]}'] = pred
    df_gen.to_csv(os.path.join(result_path, 'generate', f'opt_generate{load_epoch}.csv'), index= False)

    print(f'MIN-MAX {", ".join([f"{key}: {np.nanmin(values):.4f}-{np.nanmax(values):.4f}" for key, values in prop_dict.items()])}', flush= True)
    print(f'Average {", ".join([f"{key}: {np.nanmean(values):.4f}" for key, values in prop_dict.items()])} (elapsed time: {second2date(time.time()-s)})', flush= True)

print(f'\n---{datetime.datetime.now()}: all process done. (elapsed time: {second2date(time.time()-start)})---', flush= True)