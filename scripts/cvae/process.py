import os
import datetime
import gc
import pickle
import sys
import time
import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from utils.apps import second2date, torch_fix_seed
from utils.mask import create_mask
from utils.tree import get_pad_features
from utils.metrics import cosine_matrix
from utils.construct import reconstructMol, constructMol, calc_tanimoto


def reconstruct(dataloader,
                smiles: list,
                labels: list,
                frag_ecfps: torch.Tensor,
                ndummys: torch.Tensor,
                model: nn.Module,
                max_nfrags: int= 30, 
                useChiral: bool= True,
                free_n: bool= False,
                n_jobs: int= -1,
                device: torch.device= torch.device('cpu'),
                seed: int= 0
                ):
    torch_fix_seed(seed)

    labels = np.array(labels)
    z_list, frag_idxs_list, adjs_list = [], [], []
    label_acc, sims_list = 0, []
    s = time.time()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            frag_indices = data[0]
            features = frag_ecfps[frag_indices.flatten()].reshape(frag_indices.shape[0], frag_indices.shape[1], -1).to(device)
            positions = data[1].to(device)
            conditions = {key: data[2][:, i].float().to(device) for i, key in enumerate(model.names)}

            # make mask
            nan_mask = torch.where(data[2].isnan(), 0, 1)
            src = torch.hstack([nan_mask, torch.full((frag_indices.shape[0], 1), -1), frag_indices.detach()]).to(device)  # for super root & condition
            src_mask, _, src_pad_mask, _ = create_mask(src, src, pad_idx= 0, batch_first= True)

            # encode & decode
            z, _, _, tree_list = model(features, positions, conditions,
                                       src_mask= src_mask, src_pad_mask= src_pad_mask,
                                       frag_ecfps= frag_ecfps, ndummys= ndummys, max_nfrags= max_nfrags, free_n= free_n) 

            # stock in list
            z_list.append(z.cpu())
            frag_idxs, adjs = zip(*[(tree.dgl_graph.ndata['fid'].squeeze(-1).tolist(), tree.adjacency_matrix().tolist()) for tree in tree_list])
            frag_idxs_list += list(frag_idxs)
            adjs_list += list(adjs)

            # calc accuracy
            acc = 0
            for idxs, true_idxs in zip(frag_idxs, frag_indices):
                idxs = torch.tensor(idxs)
                idxs_pad, true_idxs_pad = pad_sequence([idxs, true_idxs], batch_first= True, padding_value= 0).chunk(2, dim= 0)
                acc += torch.mean(true_idxs_pad.eq(idxs_pad).float()).item()
            label_acc += acc / frag_indices.shape[0]

            # calc similarity of latent variables
            recon_indices = get_pad_features(tree_list, key= 'fid', max_nodes_num= max_nfrags).squeeze(-1)
            features = get_pad_features(tree_list, key= 'x', max_nodes_num= max_nfrags).to(device)
            positions = get_pad_features(tree_list, key= 'pos', max_nodes_num= max_nfrags).to(device)
            src = torch.hstack([data[2], torch.full((recon_indices.shape[0], 1), -1), recon_indices.detach()]).to(device)  # for super root & conditions
            src_mask, _, src_pad_mask, _ = create_mask(src, src, pad_idx= 0, batch_first= True)
            z_dash, _, _ = model.encode(features, positions, conditions, src_mask, src_pad_mask)
            cosines = cosine_matrix(z, z_dash).diag()
            sims_list += cosines.tolist()

            if ((i+1) % 10 == 0) | (i == 0) | ((i+1) == len(dataloader)):
                print(f'[{i+1:0=3}/{len(dataloader):0=3}] label_accuracy: {label_acc/(i+1):.6f}, latent_similarity: {cosines.mean().item():.6f}, elapsed time: {second2date(time.time()-s)}', flush= True)
                s = time.time()

    # evaluate reconstruction
    results = Parallel(n_jobs= n_jobs)(delayed(reconstructMol)(smi, labels[idxs].tolist(), adj, useChiral= useChiral) for smi, idxs, adj in zip(smiles, frag_idxs_list, adjs_list))
    dec_smiles, correct = zip(*results)
    dec_smiles = list(dec_smiles)
    correct = np.array(correct)
    if useChiral:
        print(f'reconstruction-2D rate: {sum(correct > 0)/len(smiles):.6f} ({sum(correct > 0)}/{len(smiles)})', flush= True)
        print(f'reconstruction-3D rate: {sum(correct == 3)/sum(correct > 1):.6f} ({sum(correct == 3)}/{sum(correct > 1)})', flush= True)
    else:
        print(f'reconstruction rate: {sum(correct > 0)/len(smiles):.6f} ({sum(correct > 0)}/{len(smiles)})', flush= True)
    
    # evaluate tanimoto similarity
    tanimotos = Parallel(n_jobs= n_jobs)(delayed(calc_tanimoto)(smi1, smi2, useChiral= useChiral) for smi1, smi2 in zip(smiles, dec_smiles))
    print(f'tanimoto simirality: <mean> {np.nanmean(tanimotos):.6f}, <std> {np.nanstd(tanimotos):.6f}', flush= True)

    # evaluate distribution learning
    z_all = torch.vstack(z_list)
    print(f'z-latent similarity: <mean> {np.mean(sims_list):.6f}, <std> {np.std(sims_list):.6f}', flush= True)

    return z_all.tolist(), dec_smiles, correct, tanimotos


def generate(dataloader,
             labels: list,
             frag_ecfps: torch.Tensor,
             ndummys: torch.Tensor,
             model: nn.Module,
             name_conditions: list,
             max_nfrags: int= 30, 
             useChiral: bool= True,
             free_n: bool= False,
             n_jobs: int= -1,
             device: torch.device= torch.device('cpu'),
             seed: int= 0
            ):
    torch_fix_seed(seed)

    labels = np.array(labels)
    z_list, sims_list = [], []
    frag_idxs_list, adjs_list = [], []
    s = time.time()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            z = data[0].to(device)
            conditions = {key: data[1][:, i].float().to(device) for i, key in enumerate(name_conditions)}

            # decode
            tree_list = model.sequential_decode(z, conditions, frag_ecfps, ndummys, max_nfrags= max_nfrags, free_n= free_n) 

            # calc similarity of latent variables
            recon_indices = get_pad_features(tree_list, key= 'fid', max_nodes_num= max_nfrags).squeeze(-1)
            features = get_pad_features(tree_list, key= 'x', max_nodes_num= max_nfrags).to(device)
            positions = get_pad_features(tree_list, key= 'pos', max_nodes_num= max_nfrags).to(device)
            src = torch.hstack([data[1], torch.full((recon_indices.shape[0], 1), -1), recon_indices.detach()]).to(device)  # for super root & conditions
            src_mask, _, src_pad_mask, _ = create_mask(src, src, pad_idx= 0, batch_first= True)
            z_dash, _, _ = model.encode(features, positions, conditions, src_mask, src_pad_mask)
            cosines = cosine_matrix(z, z_dash).diag()
            sims_list += cosines.tolist()

            # stock in list
            z_list.append(z_dash.cpu())
            frag_idxs, adjs = zip(*[(tree.dgl_graph.ndata['fid'].squeeze(-1).tolist(), tree.adjacency_matrix().tolist()) for tree in tree_list])
            frag_idxs_list += list(frag_idxs)
            adjs_list += list(adjs)

            if ((i+1) % 10 == 0) | (i == 0) | ((i+1) == len(dataloader)):
                print(f'[{i+1:0=3}/{len(dataloader):0=3}] latent_similarity: {cosines.mean().item():.6f}, elapsed time: {second2date(time.time()-s)}', flush= True)
                s = time.time()

    # constructMol
    dec_smiles = Parallel(n_jobs= n_jobs)(delayed(constructMol)(labels[idxs].tolist(), adj, useChiral= useChiral) for idxs, adj in zip(frag_idxs_list, adjs_list))

    z_list = torch.vstack(z_list).tolist()

    # evaluate distribution learning
    print(f'z-latent similarity: <mean> {np.mean(sims_list):.6f}, <std> {np.std(sims_list):.6f}', flush= True)
    del z, z_dash
    torch.cuda.empty_cache()

    return z_list, dec_smiles


def conditional_interpolate_around():
    pass