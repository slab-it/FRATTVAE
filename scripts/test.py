import argparse
import os

# set parser
parser = argparse.ArgumentParser(description= 'please enter paths')
parser.add_argument('yml', type= str, help= 'yml file')
parser.add_argument('--load_epoch', type= int, default= None, help= 'load model at load epoch')
parser.add_argument('--recon', action= 'store_true', help= 'only reconstruction')
parser.add_argument('--gen', action= 'store_true', help= 'only generation')

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
from torch.utils.data import Subset, DataLoader, TensorDataset
from torch.distributions.multivariate_normal import MultivariateNormal

from models.frattvae import FRATTVAE
from models.property import propLinear
from process import reconstruct, generate, CRITERION
from utils.apps import second2date
from utils.data import collate_pad_fn
from utils.preprocess import SmilesToMorganFingetPrints
from utils.chem_metrics import physchem_divergence, guacamol_fcd, get_all_metrics, METRICS_DICT

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
    frag_ecfps = torch.vstack([frag_ecfps.new_zeros(1, n_bits+dupl_bits), frag_ecfps])      # padding feature is zero vector
ndummys = torch.tensor(df_frag['ndummys'].tolist()).long()
prop_dim = sum(list(props.values())) if pnames else None
print(f'data: {data_path}', flush= True)
print(f'train: {sum(df.test==0)}, valid: {sum(df.test==-1)}, test: {sum(df.test==1)}, useChiral: {useChiral}, n_jobs: {args.n_jobs}', flush= True)
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
criterion = CRITERION[ploss]() if prop_dim else None
model.PE._update_weights()      # initialization
model.eval()


## reconstuction
if args.recon:
    s = time.time()
    print(f'---{datetime.datetime.now()}: Reconstruction start.---', flush= True)
    with open(os.path.join(result_path, 'input_data', 'dataset.pkl'), 'rb') as f:
        dataset = pickle.load(f)
    for mode, value in modes.items():
        if sum(df.test==value) == 0:
            continue
        print(f'[{mode.upper()}] data: {sum(df.test==value)}', flush= True)
        smiles = df.SMILES.loc[df.test==value].tolist()
        modedata = Subset(dataset, df.loc[df.test==value].index.tolist())
        assert len(smiles) == len(modedata)
        modeloader = DataLoader(modedata, batch_size= batch_size, shuffle= False, pin_memory= True, 
                                num_workers= 4, collate_fn= collate_pad_fn)
        z_list, dec_smiles, correct, tanimotos, pred_list = reconstruct(modeloader, smiles, uni_fragments, frag_ecfps, ndummys,
                                                                        model, pmodel, criterion, max_nfrags, useChiral, args.free_n, args.n_jobs, device)
        df_rec = pd.DataFrame({'SMILES': smiles, 'RECON': dec_smiles, 'tanimoto': tanimotos, 'correct': correct})
        if pred_list:
            for i, pred in enumerate(zip(*pred_list)):
                df_rec[f'pred{i}'] = pred

        # save results
        with open(os.path.join(result_path, mode, f'z_list{load_epoch}.pkl'), 'wb') as f:
            pickle.dump(z_list, f)
        df_rec.to_csv(os.path.join(result_path, mode, f'reconstruct{load_epoch}.csv'), index= False)

        print(f'---{datetime.datetime.now()}: {mode} done. (elapsed time: {second2date(time.time()-s)})---\n', flush= True)

    del dataset, z_list, dec_smiles, tanimotos, correct, pred_list, df_rec
    gc.collect()
    print(f'---{datetime.datetime.now()}: Reconstruction done. (elapsed time: {second2date(time.time()-s)})---\n', flush= True)


## generation
if args.gen:
    s = time.time()
    print(f'---{datetime.datetime.now()}: Generation start.---', flush= True)

    smiles_train = df.SMILES.loc[df.test==0].tolist()
    smiles_test = df.SMILES.loc[df.test==1].tolist() if (df.test==1).sum() else smiles_train

    z_mean = torch.zeros(latent_dim)
    z_var = torch.ones(latent_dim)
    dist = MultivariateNormal(z_mean, args.t * z_var * torch.eye(latent_dim))

    METRICS, METRICS_TEST = {}, {}
    for i in range(args.N):
        torch.manual_seed(i)
        z_gen = dist.sample((args.k,))
        dataloader = DataLoader(TensorDataset(z_gen), batch_size= batch_size, shuffle= False)

        z_list, dec_smiles, pred_list, cosines, euclids = generate(dataloader, uni_fragments, frag_ecfps, ndummys, 
                                                                   model, pmodel, max_nfrags, useChiral, args.free_n, args.n_jobs, device)                                    
        # eval
        properties = Parallel(n_jobs= args.n_jobs)(delayed(get_all_metrics)(s) for s in dec_smiles)
        prop_dict = {f'{key}': list(prop) for key, prop in zip(METRICS_DICT.keys(), zip(*properties))}
        df_gen = pd.DataFrame({'SMILES': dec_smiles, 'cosine': cosines, 'euclid': euclids} | prop_dict)
        if pred_list:
            for j, pred in enumerate(zip(*pred_list)):
                df_gen[f'pred{j}'] = pred

        # save results
        with open(os.path.join(result_path, 'generate', f'z_gen_list{load_epoch}_{i}.pkl'), 'wb') as f:
            pickle.dump(z_list, f)
        df_gen.to_csv(os.path.join(result_path, 'generate', f'generate{load_epoch}_{i}.csv'), index= False)

        # moses
        metrics = moses.get_all_metrics(gen= dec_smiles, k= args.k, device= device, test= smiles_train, train= smiles_train, n_jobs= args.n_jobs)
        metrics_test = moses.get_all_metrics(gen= dec_smiles, k= args.k, device= device, test= smiles_test, train= smiles_train, n_jobs= args.n_jobs)
        for key in metrics.keys():
            if i == 0:
                METRICS[key] = [metrics[key]]
                METRICS_TEST[key] = [metrics_test[key]]
            else:
                METRICS[key].append(metrics[key])
                METRICS_TEST[key].append(metrics_test[key])

        # guacamol
        dec_smiles = df_gen['SMILES'].dropna().tolist()
        if i == 0:
            METRICS['div'] = [physchem_divergence(dec_smiles, smiles_train)]
            METRICS['fcd/g'] = [guacamol_fcd(dec_smiles, smiles_train)]
            METRICS_TEST['div'] = [physchem_divergence(dec_smiles, smiles_test)]
            METRICS_TEST['fcd/g'] = [guacamol_fcd(dec_smiles, smiles_test)]
        else:
            METRICS['div'].append(physchem_divergence(dec_smiles, smiles_train))
            METRICS['fcd/g'].append(guacamol_fcd(dec_smiles, smiles_train))
            METRICS_TEST['div'].append(physchem_divergence(dec_smiles, smiles_test))
            METRICS_TEST['fcd/g'].append(guacamol_fcd(dec_smiles, smiles_test))

        print(f'[{i+1}/{args.N}] Average {", ".join([f"{key}: {np.nanmean(values):.4f}" for key, values in prop_dict.items()])} (elapsed time: {second2date(time.time()-s)})\n', flush= True)

    # print results
    print(f'moses metrics', flush= True)
    for key in METRICS.keys():
        print(f'- {key}:'.ljust(15) + f'<train> {np.nanmean(METRICS[key]):.4f} (std: {np.nanstd(METRICS[key]):.4f}), <test> {np.nanmean(METRICS_TEST[key]):.4f} (std: {np.nanstd(METRICS_TEST[key]):.4f})', flush= True)

    # save metrics
    df_metrics = pd.DataFrame(METRICS)
    df_metrics.to_csv(os.path.join(result_path, 'generate', f'metrics{load_epoch}.csv'), index= False)
    df_metrics = pd.DataFrame(METRICS_TEST)
    df_metrics.to_csv(os.path.join(result_path, 'generate', f'metrics{load_epoch}_test.csv'), index= False)

    print(f'---{datetime.datetime.now()}: Generation done. (elapsed time: {second2date(time.time()-s)})---\n', flush= True)

print(f'---{datetime.datetime.now()}: all process done. (elapsed time: {second2date(time.time()-start)})---\n', flush= True)