import argparse
import os

# set parser
parser = argparse.ArgumentParser(description= 'please enter paths')
parser.add_argument('yml', type= str, help= 'yml file')
parser.add_argument('--gpu', type= int, default= 0, help= 'gpu device id')
parser.add_argument('--n_jobs', type= int, default= 1, help= 'the number of cpu for parallel, default 1')
parser.add_argument('--load_epoch', type= int, default= 0, help= 'load model at load epoch, default epoch= 0')
parser.add_argument('--save_interval', type= int, default= 20, help= 'save model every N epochs, default N= 20')
parser.add_argument('--select_metric', type= int, default= 4, help= 'metric to select best model. 0: total, 1: kl, 2: label loss, 3: prop loss, 4: label acc, default 4')
parser.add_argument('--freeze_params', action= 'store_true', help= 'freeze parameters. the parameters of last layers are always learning.')
parser.add_argument('--freeze_enc_params', action= 'store_true', help= 'freeze encoder parameters.')
parser.add_argument('--replay', action= 'store_true')
parser.add_argument('--valid', action= 'store_true')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import datetime
import gc
import glob
import pickle
import time
import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import yaml
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader

from models.frattvae import FRATTVAE
from models.property import propLinear
from models.wrapper import PropWrapper
from utils.apps import second2date, torch_fix_seed, list2pdData
from utils.data import collate_pad_fn
from utils.mask import create_mask
from utils.metrics import batched_kl_divergence, CRITERION
from utils.preprocess import SmilesToMorganFingetPrints

yml_file = args.yml
load_epoch = args.load_epoch
# yml_file = ''

start = time.time()
print(f'---{datetime.datetime.now()}: start.---', flush= True)

## check environments
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'GPU [{args.gpu}] is available: {torch.cuda.is_available()}', flush= True)

## load hyperparameters
with open(yml_file) as yml:
    params = yaml.safe_load(yml)
torch_fix_seed(params['seed'])
print(f'load: {yml_file}', flush= True)

# path
result_path = params['result_path']
data_path = params['data_path']
model_path = params['model_path'] if load_epoch == 0 else os.path.join(result_path, 'models', f'model_iter{load_epoch}.pth')
pmodel_path = params['pmodel_path'] if (load_epoch == 0) | (params['pmodel_path'] is None) else os.path.join(result_path, 'models', f'pmodel_iter{load_epoch}.pth')
frag_path = params['frag_path']
base_path = params['base_path']
with open(os.path.join(base_path, 'input_data', 'params.yml')) as yml:
    base_params = yaml.safe_load(yml)
print(f"base load: {os.path.join(base_path, 'input_data', 'params.yml')}\n", flush= True)

# hyperparameters for decomposition and tree-fragments
decomp_params = params['decomp']
min_size = decomp_params['min_size']
max_nfrags = decomp_params['max_nfrags']
n_bits = decomp_params['n_bits']
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
ploss = model_params['ploss']
dropout = model_params['dropout']
props = model_params['property']
pnames = list(props.keys())

# hyperparameters for training
train_params = params['train']
epochs = train_params['epoch']
batch_size = train_params['batch_size'] // 2 if args.replay else train_params['batch_size']
lr = train_params['lr']
p_lr = train_params['p_lr']
kl_w = train_params['kl_w']
l_w = train_params['l_w']
p_w = train_params['p_w']
reset_prop = train_params['reset']

## load data
print(f'---{datetime.datetime.now()}: Loading data. ---', flush= True)
s = time.time()

# add data
df = pd.read_csv(data_path)
smiles = df.SMILES.tolist()
with open(os.path.join(result_path, 'input_data', 'dataset.pkl'), 'rb') as f:
    dataset = pickle.load(f)
add_data = Subset(dataset, df.loc[df.test==0].index.tolist())
valid_data = Subset(dataset, df.loc[df.test==-1].index.tolist()) if args.valid & np.any(df.test==-1) else None

# replay data
if args.replay:
    df_replay = pd.read_csv(base_params['data_path'])
    try:
        replay_path = os.path.join(base_path, 'input_data', 'dataset.pkl')
        with open(replay_path, 'rb') as f:
            dataset = pickle.load(f)
        replay_data = Subset(dataset, df_replay.loc[df_replay.test==0].index.tolist())
    except:
        replay_path = glob.glob(os.path.join(base_path, 'input_data', 'dataset_train*.pkl'))[0]
        with open(replay_path, 'rb') as f:
            replay_data = pickle.load(f)

# fragment to ECFP
df_frag = pd.read_csv(frag_path)
uni_fragments = df_frag.SMILES.tolist()
freq_label = df_frag['frequency'].tolist()
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
num_labels= frag_ecfps.shape[0] if load_epoch > 0 else len(pd.read_csv(base_params['frag_path']))
prop_dim = sum(list(props.values())) if pnames else None
print(f'add data: {data_path}', flush= True)
print(f'train: {sum(df.test==0)}, valid: {sum(df.test==-1)}, test: {sum(df.test==1)}, useChiral: {useChiral}, n_jobs: {args.n_jobs}', flush= True)
if args.replay:
    print(f'replay data: {replay_path}, train: {len(replay_data)}', flush= True)
print(f'fragments: {len(uni_fragments)}, feature: {frag_ecfps.shape[-1]}, tree: ({max_depth}, {max_degree}), prop: {prop_dim}\n', flush= True)

# make data loader
add_loader = DataLoader(add_data, batch_size= batch_size, shuffle= True, num_workers= 4,
                        pin_memory= True, collate_fn= collate_pad_fn)
if args.replay:
    replay_loader = DataLoader(replay_data, batch_size= batch_size, shuffle= True, num_workers= 4,
                               pin_memory= True, collate_fn= collate_pad_fn)
valid = bool(valid_data)
if valid:
    valid_loader = DataLoader(valid_data, batch_size= batch_size, shuffle= False, num_workers= 4,
                              pin_memory= True, collate_fn= collate_pad_fn)

# define model
model = FRATTVAE(num_labels, max_depth, max_degree, feat_dim, latent_dim, 
               d_model, d_ff, num_layers, num_heads, activation, dropout).to(device)
model.load_state_dict(torch.load(model_path, map_location= device))
model.PE._update_weights()      # initialization

# freeze parameters
if args.freeze_params | args.freeze_enc_params:
    for para in model.parameters():
        para.requires_grad = False
    for para in model.fc_dec.parameters():
        para.requires_grad = True
    if args.freeze_enc_params:
        for para in model.decoder.parameters():
            para.requires_grad = True

# define model for property prediction
if prop_dim:
    pmodel = propLinear(latent_dim, prop_dim).to(device)
    if pmodel_path:
        pmodel.load_state_dict(torch.load(pmodel_path), map_location= device)
else:
    pmodel = None
model = PropWrapper(model, pmodel).to(device)
print(f'model loaded: {model_path}', flush= True)

# class increment
if num_labels != frag_ecfps.shape[0]:
    print(f'the last layer changed: {num_labels} -> {frag_ecfps.shape[0]}', flush= True)
    num_labels = frag_ecfps.shape[0]
    model.vae.fc_dec = nn.Linear(d_model, num_labels).to(device)
    
# define optimizer
if pmodel:
    optimizer = optim.Adam([{'params': model.vae.parameters()},
                            {'params': model.pmodel.parameters(), 'lr': p_lr}], lr= lr, eps= 1e-3)
else:
    optimizer = optim.Adam(model.parameters(), lr= lr, eps= 1e-3)

# define loss
freq_label = torch.tensor(freq_label)
freq_label[freq_label > 1000] = 1000                              # limitation
loss_weight_label = freq_label.max() / freq_label
loss_weight_label[loss_weight_label == float('Inf')]  = 0.001     # padding weight
loss_weight_label = loss_weight_label.to(device) if loss_weight_label is not None else None
crirerion = nn.CrossEntropyLoss(weight= loss_weight_label)

prop = df[pnames].to_numpy()
if (ploss == 'bce') & np.all(~np.isnan(prop)):
    loss_weight_prop = torch.tensor((prop==0).sum(axis= 0) / prop.sum(axis= 0)).float().to(device)
    pcriterion = CRITERION[ploss](pos_weight= loss_weight_prop)
else:
    loss_weight_prop = 0
    pcriterion = CRITERION[ploss]()
pcriterion = CRITERION[ploss]()

# release memory
del add_data, valid_data, dataset, prop
del df, df_frag, uni_fragments, freq_label, loss_weight_label, loss_weight_prop
if args.replay: del replay_data, df_replay
gc.collect()
print(f'---{datetime.datetime.now()}: Loading data done. (elapsed time: {second2date(time.time()-s)})---\n', flush= True)


## finetune
print(f'---{datetime.datetime.now()}: Finetuning start (valid: {args.valid}, replay: {args.replay}).---', flush= True)
print(f'epoch: {epochs}, batch_size: {batch_size}, learning_rate: [{lr}, {p_lr}], dropout: {dropout}, save: {args.save_interval}, kl_w: {kl_w}, l_w: {l_w}, p_w: {p_w:.2f}', flush= True)
s = s_dash = time.time()

filename = os.path.join(result_path, 'train', f'train_{datetime.date.today()}.txt')
with open(filename, 'a') as f:
    f.write(f'---{datetime.datetime.now()}: Finetuning start.---\n')

metrics = ['total', 'kl', 'label', 'prop', 'label_acc', 'label_pad_acc']
TRAIN_LOSS, VALID_LOSS = [], []
metric = args.select_metric
before = 0 if metric > 3 else float('-inf')
patient = 0
epochs = load_epoch + epochs
for epoch in range(load_epoch, epochs):
    model.train()
    train_losses = []

    if args.replay: replay_iterator = replay_loader.__iter__()
    for i, data in enumerate(add_loader):
        optimizer.zero_grad()
        
        if args.replay:
             # concat add_data and replay_data
            replay_data = replay_iterator.next()
            frag1, pos1 = data[0], data[1]
            frag2, pos2 = replay_data[0], replay_data[1]
            if frag1.shape[-1] != frag2.shape[-1]:
                extra_pad = frag1.shape[-1] - frag2.shape[-1]
                frag1 = torch.hstack([frag1, frag1.new_zeros(frag1.shape[0], abs(extra_pad))]) if extra_pad < 0 else frag1
                frag2 = torch.hstack([frag2, frag2.new_zeros(frag2.shape[0], abs(extra_pad))]) if extra_pad > 0 else frag2
                pos1 = torch.cat([pos1, pos1.new_zeros(pos1.shape[0], abs(extra_pad), pos1.shape[-1])], dim= 1) if extra_pad < 0 else pos1
                pos2 = torch.cat([pos2, pos2.new_zeros(pos2.shape[0], abs(extra_pad), pos2.shape[-1])], dim= 1) if extra_pad > 0 else pos2
            frag_indices = torch.vstack([frag1, frag2])
            features = frag_ecfps[frag_indices.flatten()].reshape(frag_indices.shape[0], frag_indices.shape[1], -1).to(device)
            positions = torch.vstack([pos1, pos2]).to(device)
            if reset_prop:
                prop = torch.vstack([data[2], torch.full(size= (replay_data[2].shape[0], data[2].shape[-1]), fill_value= float('nan'))]).to(device)
            else:
                prop = torch.vstack([data[2], replay_data[2]]).to(device)
        else:
            frag_indices = data[0]
            features = frag_ecfps[frag_indices.flatten()].reshape(frag_indices.shape[0], frag_indices.shape[1], -1).to(device)
            positions = data[1].to(device)
            prop = data[2].to(device)
        target = torch.hstack([frag_indices.detach(), torch.zeros(frag_indices.shape[0], 1)]).flatten().long().to(device)

        # make mask
        frag_indices = torch.hstack([torch.full((frag_indices.shape[0], 1), -1), frag_indices.detach()]).to(device)  # for super root
        src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(frag_indices, frag_indices, pad_idx= 0, batch_first= True)

        # forward
        z, mu, ln_var, output, pred_prop = model(features, positions, 
                                                 src_mask, src_pad_mask, 
                                                 tgt_mask, tgt_pad_mask)     # output: shape= (B, L+1, num_labels)
        
        # property prediction
        prop_mask = ~torch.isnan(prop)
        if torch.all(~prop_mask) | (pmodel is None):
            prop_loss = torch.tensor(0).to(device)
            p_f = float('nan')
        elif torch.all(prop_mask):
            prop_loss = pcriterion(input= pred_prop, target= prop)
            p_f = 1
        else:
            prop_loss = pcriterion(input= pred_prop[prop_mask], target= prop[prop_mask])
            p_f = 1
            
        # calc loss
        kl_loss = batched_kl_divergence(mu, ln_var)
        label_loss = crirerion(input= output.view(-1, num_labels), target= target)

        # backward
        total_loss = kl_w * kl_loss + l_w * label_loss + p_w * prop_loss
        total_loss.backward()
        optimizer.step()

        # calc accuracy
        equals = output.argmax(dim= -1).flatten().eq(target)
        label_acc = equals[target!=0].sum() / (target!=0).sum()
        label_pad_acc = equals.sum() / target.shape[0]

        train_losses.append([total_loss.item(), kl_loss.item(), label_loss.item(), p_f * prop_loss.item(), label_acc.item(), label_pad_acc.item()])
        # print(f'<{i+1:0=3}/{iters:0=3}> ' + ', '.join([f'{m}: {l:.4f}' for m, l in zip(metrics, train_losses[-1])]) + f', elapsed time: {second2date(time.time()-s)}', flush= True)

    TRAIN_LOSS.append([np.nanmean(losses) for losses in zip(*train_losses)])
    with open(filename, 'a') as f:
        f.write(f'[{epoch+1:0=3}/{epochs:0=3}] <train> ' + ', '.join([f'{m}: {l:.4f}' for m, l in zip(metrics, TRAIN_LOSS[-1])]) + f', elapsed time: {second2date(time.time()-s)}\n')
    if (epoch == 0) | ((epoch+1) % 5 == 0):
        print(f'[{epoch+1:0=3}/{epochs:0=3}] <train> ' + ', '.join([f'{m}: {l:.4f}' for m, l in zip(metrics, TRAIN_LOSS[-1])]) + f', elapsed time: {second2date(time.time()-s)}', flush= True)
    s = time.time()

    # validation
    if valid:
        model.eval()
        valid_losses = []
        with torch.no_grad():
            for data in valid_loader:
                frag_indices = data[0]
                features = frag_ecfps[frag_indices.flatten()].reshape(frag_indices.shape[0], frag_indices.shape[1], -1).to(device)
                positions = data[1].to(device)
                prop = data[2].to(device)
                target = torch.hstack([frag_indices.detach(), torch.zeros(frag_indices.shape[0], 1)]).flatten().long().to(device)

                # make mask
                frag_indices = torch.hstack([torch.full((frag_indices.shape[0], 1), -1), frag_indices.detach()]).to(device)  # for super root
                src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(frag_indices, frag_indices, pad_idx= 0, batch_first= True)

                # forward
                z, mu, ln_var, output, pred_prop = model(features, positions, 
                                                         src_mask, src_pad_mask, 
                                                         tgt_mask, tgt_pad_mask)     # output: shape= (B, L+1, num_labels)
                
                # property prediction
                prop_mask = ~torch.isnan(prop)
                if torch.all(~prop_mask) | (pmodel is None):
                    prop_loss = torch.tensor(0).to(device)
                    p_f = float('nan')
                elif torch.all(prop_mask):
                    prop_loss = pcriterion(input= pred_prop, target= prop)
                    p_f = 1
                else:
                    prop_loss = pcriterion(input= pred_prop[prop_mask], target= prop[prop_mask])
                    p_f = 1

                # calc loss
                kl_loss = batched_kl_divergence(mu, ln_var)
                label_loss = crirerion(input= output.view(-1, num_labels), target= target)
                total_loss = kl_w * kl_loss + l_w * label_loss + p_w * prop_loss
                
                # calc accuracy
                equals = output.argmax(dim= -1).flatten().eq(target)
                label_acc = equals[target!=0].sum() / (target!=0).sum()
                label_pad_acc = equals.sum() / target.shape[0]

                valid_losses.append([total_loss.item(), kl_loss.item(), label_loss.item(), p_f * prop_loss.item(), label_acc.item(), label_pad_acc.item()])

        VALID_LOSS.append([np.nanmean(losses) for losses in zip(*valid_losses)])
        with open(filename, 'a') as f:
            f.write(f'[{epoch+1:0=3}/{epochs:0=3}] <valid> ' + ', '.join([f'{m}: {l:.4f}' for m, l in zip(metrics, VALID_LOSS[-1])]) + f', elapsed time: {second2date(time.time()-s)}\n')
        if (epoch == 0) | ((epoch+1) % 5 == 0):
            print(f'[{epoch+1:0=3}/{epochs:0=3}] <valid> ' + ', '.join([f'{m}: {l:.4f}' for m, l in zip(metrics, VALID_LOSS[-1])]) + f', elapsed time: {second2date(time.time()-s)}', flush= True)
        
    # save models
    if ((epoch+1) % args.save_interval == 0) | ((epoch+1) == epochs):
        torch.save(model.vae.state_dict(), os.path.join(result_path, 'models', f'model_iter{epoch+1}.pth'))
        if prop_dim: torch.save(model.pmodel.state_dict(), os.path.join(result_path, 'models', f'pmodel_iter{epoch+1}.pth'))
        torch.save(optimizer.state_dict(), os.path.join(result_path, 'models', f'optim_iter{epoch+1}.pth'))
        print(f'model_iter{epoch+1} saved.', flush= True)

    # best model save n= 0: total, 1: kl, 2: label loss, 3: prop loss, 4: label_acc
    tmp = VALID_LOSS[-1][metric] if valid else TRAIN_LOSS[-1][metric]
    if metric < 4:
        tmp = -1 * tmp
    if tmp > before:
        ep = epoch + 1
        before = tmp
        best_state_dict = deepcopy(model.vae.state_dict())
        if prop_dim: best_state_pdict = deepcopy(model.pmodel.state_dict())
        patient = 0
    else:
        patient += 1

    if (patient > 5) | ((epoch+1) == epochs):
        torch.save(best_state_dict, os.path.join(result_path, 'models', f'model_best.pth'))
        if prop_dim: torch.save(best_state_pdict, os.path.join(result_path, 'models', f'pmodel_best.pth'))
        print(f'model_iter{ep} saved. [{metrics[metric]}: {abs(before):.4f}]', flush= True)
        patient = 0

# save loss and reconstruction
with open(filename, 'a') as f:
    f.write(f'---{datetime.datetime.now()}: Finetuning done.---\n')
TRAIN_LOSS = list(zip(*TRAIN_LOSS))
df_loss = list2pdData(TRAIN_LOSS, metrics)
df_loss.to_csv(os.path.join(result_path, 'train', f'loss{load_epoch}-{epochs}.csv'), index= False)
if VALID_LOSS:
    VALID_LOSS = list(zip(*VALID_LOSS))
    df_loss = list2pdData(VALID_LOSS, metrics)
    df_loss.to_csv(os.path.join(result_path, 'valid', f'loss{load_epoch}-{epochs}.csv'), index= False)
print(f'---{datetime.datetime.now()}: Finetuning done. (elapsed time: {second2date(time.time()-s_dash)})---\n', flush= True)

print(f'---{datetime.datetime.now()}: all process done. (elapsed time: {second2date(time.time()-start)})---', flush= True)