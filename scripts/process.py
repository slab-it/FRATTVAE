import os
import datetime
import gc
import pickle
import random
import time
import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from utils.apps import second2date, torch_fix_seed
from utils.mask import create_mask
from utils.tree import get_pad_features, get_tree_features
from utils.metrics import cosine_matrix, euclid_distance, CRITERION
from utils.preprocess import debugMolToBRICSfragments
from utils.construct import reconstructMol, constructMol, calc_tanimoto


def reconstruct(dataloader,
                smiles: list,
                labels: list,
                frag_ecfps: torch.Tensor,
                ndummys: torch.Tensor,
                model: nn.Module,
                pmodel: nn.Module= None,
                criterion= None,
                max_nfrags: int= 30, 
                useChiral: bool= True,
                free_n: bool= False,
                n_jobs: int= -1,
                device: torch.device= torch.device('cpu'),
                seed: int= 0
                ):
    torch_fix_seed(seed)

    labels = np.array(labels)
    z_list, prop_list, pred_list = [], [], []
    frag_idxs_list, adjs_list = [], []
    label_acc, sims_list = 0, []
    s = time.time()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            frag_indices = data[0]
            features = frag_ecfps[frag_indices.flatten()].reshape(frag_indices.shape[0], frag_indices.shape[1], -1).to(device)
            positions = data[1].to(device)
            prop = data[2].to(device)

            # make mask
            src = torch.hstack([torch.full((frag_indices.shape[0], 1), -1), frag_indices.detach()]).to(device)  # for super root
            src_mask, _, src_pad_mask, _ = create_mask(src, src, pad_idx= 0, batch_first= True)

            # encode & decode
            z, _, _, tree_list = model(features, positions,
                                       src_mask= src_mask, src_pad_mask= src_pad_mask,
                                       frag_ecfps= frag_ecfps, ndummys= ndummys, max_nfrags= max_nfrags, free_n= free_n) 

            # property prediction
            if pmodel:
                pred_prop = pmodel(z.to(device))
                prop_list.append(prop)
                pred_list.append(pred_prop)

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
            src = torch.hstack([torch.full((recon_indices.shape[0], 1), -1), recon_indices.detach()]).to(device)  # for super root
            src_mask, _, src_pad_mask, _ = create_mask(src, src, pad_idx= 0, batch_first= True)
            z_dash, _, _ = model.encode(features, positions, src_mask, src_pad_mask)
            cosines = cosine_matrix(z, z_dash).diag()
            sims_list += cosines.tolist()

            if ((i+1) % 10 == 0) | (i == 0) | ((i+1) == len(dataloader)):
                print(f'[{i+1:0=3}/{len(dataloader):0=3}] label_accuracy: {label_acc/(i+1):.6f}, cosine_similarity: {cosines.mean().item():.6f}, elapsed time: {second2date(time.time()-s)}', flush= True)
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

    # evaluate property prediction
    if pmodel:
        ploss_list = []
        prop = torch.vstack(prop_list)
        pred = torch.vstack(pred_list)
        assert prop.shape[0] == pred.shape[0]
        prop_dim = prop.shape[-1]
        with torch.no_grad():
            for i in range(prop_dim):
                mask = ~torch.isnan(prop[:, i])
                ploss_list.append(criterion(input= pred[:, i][mask], target= prop[:, i][mask]).item())
        pred_list = pred.tolist()
        print(f'property prediction: ' + ', '.join([f'[{i}] {pl:.4f}' for i, pl in enumerate(ploss_list)]), flush= True)

    return z_all.tolist(), dec_smiles, correct, tanimotos, pred_list


def generate(dataloader,
             labels: list,
             frag_ecfps: torch.Tensor,
             ndummys: torch.Tensor,
             model: nn.Module,
             pmodel: nn.Module= None,
             max_nfrags: int= 30, 
             useChiral: bool= True,
             free_n: bool= False,
             n_jobs: int= -1,
             device: torch.device= torch.device('cpu'),
             seed: int= 0
            ):
    torch_fix_seed(seed)

    labels = np.array(labels)
    z_list, pred_list, cosine_list, rmse_list = [], [], [], []
    frag_idxs_list, adjs_list = [], []
    s = time.time()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            z = data[0].to(device)

            # decode
            tree_list = model.sequential_decode(z, frag_ecfps, ndummys, max_nfrags= max_nfrags, free_n= free_n) 

            # property prediction
            if pmodel:
                pred_prop = pmodel(z.to(device))
                pred_list.append(pred_prop)

            # calc similarity of latent variables
            recon_indices = get_pad_features(tree_list, key= 'fid', max_nodes_num= max_nfrags).squeeze(-1)
            features = get_pad_features(tree_list, key= 'x', max_nodes_num= max_nfrags).to(device)
            positions = get_pad_features(tree_list, key= 'pos', max_nodes_num= max_nfrags).to(device)
            src = torch.hstack([torch.full((recon_indices.shape[0], 1), -1), recon_indices.detach()]).to(device)  # for super root
            src_mask, _, src_pad_mask, _ = create_mask(src, src, pad_idx= 0, batch_first= True)
            z_dash, _, _ = model.encode(features, positions, src_mask, src_pad_mask)
            cosines = cosine_matrix(z, z_dash).diag()
            rmses = euclid_distance(z, z_dash) / np.sqrt(model.latent_dim)

            # stock in list
            z_list.append(z_dash.cpu())
            frag_idxs, adjs = zip(*[(tree.dgl_graph.ndata['fid'].squeeze(-1).tolist(), tree.adjacency_matrix().tolist()) for tree in tree_list])
            frag_idxs_list += list(frag_idxs)
            adjs_list += list(adjs)
            cosine_list += cosines.tolist()
            rmse_list += rmses.tolist()

            if ((i+1) % 10 == 0) | (i == 0) | ((i+1) == len(dataloader)):
                print(f'[{i+1:0=3}/{len(dataloader):0=3}] cosine_similarity: {cosines.mean().item():.4f}, RMSE: {rmses.mean().item():.4f}, elapsed time: {second2date(time.time()-s)}', flush= True)
                s = time.time()

    # constructMol
    dec_smiles = Parallel(n_jobs= n_jobs)(delayed(constructMol)(labels[idxs].tolist(), adj, useChiral= useChiral) for idxs, adj in zip(frag_idxs_list, adjs_list))

    z_list = torch.vstack(z_list).tolist()
    if pmodel:
        pred_list = torch.vstack(pred_list).tolist()

    # evaluate distribution learning
    print(f'z-latent similarity:', flush= True)
    print(f'- cosine:  <mean> {np.mean(cosine_list):.6f}, <std> {np.std(cosine_list):.6f}', flush= True)
    print(f'- RMSE:  <mean> {np.mean(rmse_list):.6f}, <std> {np.std(rmse_list):.6f}', flush= True)
    del z, z_dash
    torch.cuda.empty_cache()

    return z_list, dec_smiles, pred_list, cosine_list, rmse_list


def interpolate_between_2points(mol0,
                                mol1,
                                model: nn.Module,
                                labels: list,
                                frag_ecfps: torch.Tensor,
                                ndummys: torch.Tensor,
                                min_size: int= 1,
                                max_nfrags: int= 30, 
                                useChiral: bool= True,
                                ignore_double: bool= True,
                                num_intp: int= 100,
                                free_n: bool= False,
                                n_jobs: int= -1,
                                device: torch.device= torch.device('cpu'),
                                ):
    # convert mol to latent variable and finger print
    tmp = []
    for mol in [mol0, mol1]:
        frags, bond_types, bondMapNums, _ = debugMolToBRICSfragments(mol, minFragSize= min_size, maxFragNums= max_nfrags, 
                                                                     maxDegree= model.width, useChiral= useChiral, ignore_double= ignore_double)
        frag_idxs = [labels.index(f) for f in frags]
        frag_idxs, features, positions = get_tree_features(frag_idxs, frag_ecfps[frag_idxs], bond_types, bondMapNums, model.depth, model.width, free_n)
        frag_idxs, features, positions = frag_idxs.unsqueeze(0), features.unsqueeze(0), positions.unsqueeze(0)
        src = torch.hstack([torch.full((frag_idxs.shape[0], 1), -1), frag_idxs.detach()])
        src_mask, _, src_pad_mask, _ = create_mask(src, src, pad_idx= 0, batch_first= True)
        z, _, _ = model.encode(features.to(device), positions.to(device), src_mask.to(device), src_pad_mask.to(device))
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048, useChirality= useChiral)
        tmp += [z, fp]
    z0, fp0, z1, fp1 = tmp

    # interpolate
    labels = np.array(labels)
    d = z1 - z0
    similarity = DataStructs.TanimotoSimilarity(fp0, fp1)
    z_intp = torch.vstack([z0 + ((i*d)/(num_intp+1)) for i in range(1, num_intp+1)]).float()

    with torch.no_grad():
        tree_list = model.sequential_decode(z_intp.to(device), frag_ecfps, ndummys, max_nfrags= max_nfrags)
    frag_idxs_list, adjs_list = map(list, zip(*[(tree.dgl_graph.ndata['fid'].squeeze(-1).tolist(), tree.adjacency_matrix().tolist()) for tree in tree_list]))
    smiles_intp = Parallel(n_jobs= n_jobs)(delayed(constructMol)(labels[idxs].tolist(), adj, useChiral= useChiral) for idxs, adj in zip(frag_idxs_list, adjs_list))
    mols_intp = [Chem.MolFromSmiles(s) for s in smiles_intp]
    similarity_intp = [DataStructs.TanimotoSimilarity(fp0, AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048, useChirality= useChiral)) for m in mols_intp]
    
    return smiles_intp, similarity_intp, similarity


def eval_interpolate_between_2points(smiles: list,
                                     model: nn.Module,
                                     labels: list,
                                     frag_ecfps: torch.Tensor,
                                     ndummys: torch.Tensor,
                                     min_size: int= 1,
                                     max_nfrags: int= 30, 
                                     useChiral: bool= True,
                                     ignore_double: bool= False,
                                     num_intp: int= 100,
                                     num_iters: int= 10000,
                                     free_n: bool= False,
                                     n_jobs: int= -1,
                                     device: torch.device= torch.device('cpu'),
                                     seed: int= 0,
                                     ):
    """
    Evaluate smoothness of latent space quantitatively
    """
    torch_fix_seed(seed)
    sample_list, unique_list, mae_list, corr_list = [], [], [], []
    s = time.time()
    for i in range(num_iters):
        random.shuffle(smiles)
        s0, s1 = smiles[:2]
        smiles_intp, similarity_intp, similarity = interpolate_between_2points(Chem.MolFromSmiles(s0), Chem.MolFromSmiles(s1), model, labels, frag_ecfps, ndummys, 
                                                                               min_size, max_nfrags, useChiral, ignore_double, 
                                                                               num_intp, free_n, n_jobs, device)
        d_sim = 1 - similarity
        similarity_intp = torch.tensor(similarity_intp)
        ideal_similarity_intp = torch.tensor([1 - (i*d_sim)/(num_intp+1) for i in range(1, num_intp+1)]).float()

        sample_list.append((s0, s1, similarity))
        unique_list.append(len(set([s for s in smiles_intp if s is not None]))/num_intp)
        mae_list.append(F.l1_loss(similarity_intp, ideal_similarity_intp).item())
        corr_list.append(torch.corrcoef(torch.vstack([similarity_intp, ideal_similarity_intp]))[0, 1].item())

        if ((i+1) % 100 == 0):
            print(f'[{i+1}/{num_iters}] unique: {np.mean(unique_list):.4f}, R: {np.mean(corr_list):.4f}, MAE: {np.mean(mae_list):.4f}, elapsed time: {second2date(time.time()-s)}', flush= True)
            s = time.time()

    print(f'[{i+1}/{num_iters}] unique: {np.mean(unique_list):.4f} (std; {np.std(unique_list):.4f}), R: {np.mean(corr_list):.4f} (std; {np.std(corr_list):.4f}),',
           f'MAE: {np.mean(mae_list):.4f} (std; {np.std(mae_list):.4f}), elapsed time: {second2date(time.time()-s)}', flush= True)

    return sample_list, unique_list, mae_list, corr_list


def interpolate_around(mol,
                       model: nn.Module,
                       labels: list,
                       frag_ecfps: torch.Tensor,
                       ndummys: torch.Tensor,
                       min_size: int= 1,
                       max_nfrags: int= 30, 
                       useChiral: bool= True,
                       ignore_double: bool= False,
                       radius: int= 4,
                       delta: int= 5,
                       free_n: bool= False,
                       n_jobs: int= -1,
                       device: torch.device= torch.device('cpu'),
                       seed: int= 0
                        ):
    """
    num_intp_per_axis: odd number
    """
    torch_fix_seed(seed)

    # convert mol to latent variable and finger print
    frags, bond_types, bondMapNums, _ = debugMolToBRICSfragments(mol, minFragSize= min_size, maxFragNums= max_nfrags, 
                                                                 maxDegree= model.width, useChiral= useChiral, ignore_double= ignore_double)
    frag_idxs = [labels.index(f) for f in frags]
    frag_idxs, features, positions = get_tree_features(frag_idxs, frag_ecfps[frag_idxs], bond_types, bondMapNums, model.depth, model.width, free_n)
    frag_idxs, features, positions = frag_idxs.unsqueeze(0), features.unsqueeze(0), positions.unsqueeze(0)
    src = torch.hstack([torch.full((frag_idxs.shape[0], 1), -1), frag_idxs.detach()])
    src_mask, _, src_pad_mask, _ = create_mask(src, src, pad_idx= 0, batch_first= True)
    z, _, _ = model.encode(features.to(device), positions.to(device), src_mask.to(device), src_pad_mask.to(device))
    z = z.cpu()
    labels = np.array(labels)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048, useChirality= useChiral)

    # randomly generate 2 orthonormal axis x & y.
    x = np.random.randn(z.shape[-1])
    x /= np.linalg.norm(x)

    y = np.random.randn(z.shape[-1])
    y -= y.dot(x) * x
    y /= np.linalg.norm(y)

    x, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()

    z_intp = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            z_intp.append(z + x * delta * dx + y * delta * dy)
            if (dx == 0) & (dy == 0):
                center = len(z_intp) - 1
    z_intp = torch.vstack(z_intp).float()

    with torch.no_grad():
        tree_list = model.sequential_decode(z_intp.to(device), frag_ecfps, ndummys, max_nfrags= max_nfrags)
    frag_idxs_list, adjs_list = map(list, zip(*[(tree.dgl_graph.ndata['fid'].squeeze(-1).tolist(), tree.adjacency_matrix().tolist()) for tree in tree_list]))
    smiles_intp = Parallel(n_jobs= n_jobs)(delayed(constructMol)(labels[idxs].tolist(), adj, useChiral= useChiral) for idxs, adj in zip(frag_idxs_list, adjs_list))
    mols_intp = [Chem.MolFromSmiles(s) for s in smiles_intp]
    # fp = AllChem.GetMorganFingerprintAsBitVect(mols_intp[center], 2, 2048, useChirality= useChiral)
    similarity_intp = [DataStructs.TanimotoSimilarity(fp, AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048, useChirality= useChiral)) for m in mols_intp]
    
    return smiles_intp, similarity_intp


def eval_interpolate_around(smiles: list,
                            model: nn.Module,
                            labels: list,
                            frag_ecfps: torch.Tensor,
                            ndummys: torch.Tensor,
                            min_size: int= 1,
                            max_nfrags: int= 30, 
                            useChiral: bool= True,
                            ignore_double: bool= False,
                            radius: int= 4,
                            delta: int= 5,
                            num_iters: int= 10000,
                            free_n: bool= False,
                            n_jobs: int= -1,
                            device: torch.device= torch.device('cpu'),
                            seed: int= 0,
                            ):
    """
    Evaluate smoothness of latent space quantitatively
    """
    torch_fix_seed(seed)
    sample_list, unique_dict, similarity_dict = [], {}, {}
    s = time.time()
    random.shuffle(smiles)
    for i in range(num_iters):
        s0 = smiles[i]
        sample_list.append(s0)
        for j in range(1, radius+1):
            if i == 0:
                unique_dict[f'unique:{delta*j}'] = []
                similarity_dict[f'similarity:{delta*j}'] = []
            smiles_intp, similarity_intp = interpolate_around(Chem.MolFromSmiles(s0), model, labels, frag_ecfps, ndummys, 
                                                              min_size, max_nfrags, useChiral, ignore_double, 
                                                              1, delta*j, free_n, n_jobs, device, seed)
            unique_dict[f'unique:{delta*j}'].append((len(set([s for s in smiles_intp if s is not None]))-1) / (len(smiles_intp)-1))
            similarity_dict[f'similarity:{delta*j}'].append((np.sum(similarity_intp)-1)/(len(similarity_intp)-1))

        if ((i+1) % 100 == 0):
            print(f'[{i+1}/{num_iters}] unique, similarity, elapsed time: {second2date(time.time()-s)}', flush= True)
            for j in range(1, radius+1):
                print(f'- delta[{delta*j}]: {np.mean(unique_dict[f"unique:{delta*j}"]):.4f}, {np.mean(similarity_dict[f"similarity:{delta*j}"]):.4f}', flush= True)
            s = time.time()
        
    print(f'[{i+1}/{num_iters}] unique, similarity, elapsed time: {second2date(time.time()-s)}', flush= True)
    for j in range(1, radius+1):
        print(f'- delta[{delta*j}]: {np.mean(unique_dict[f"unique:{delta*j}"]):.4f} (std; {np.std(unique_dict[f"unique:{delta*j}"]):.4f}),',
              f'{np.mean(similarity_dict[f"similarity:{delta*j}"]):.4f} (std; {np.std(similarity_dict[f"similarity:{delta*j}"]):.4f})', flush= True)

    return sample_list, unique_dict, similarity_dict


def prop_optimize(z: torch.Tensor, 
                  target: torch.Tensor, 
                  pmodel: nn.Module, 
                  criterion: nn.Module= nn.MSELoss(reduction= 'none'),
                  lr: float= 0.01,
                  max_iter: int= 100,
                  max_patient: int= 5,
                  device: torch.device= torch.device('cpu')
                  ) -> torch.Tensor:
    """
    optimize latent-variables using gradient.
    z: torch.Tensor, shape= (batch_size, z_dim)
    target: torch.Tensor, shape= (batch_size, prop_dim)
    """
    z_opt = z.detach().clone().to(device)
    z_opt.requires_grad = True
    target = target.to(device)
    optimizer = torch.optim.Adam([z_opt], lr= lr)
    best_z, best_loss = z_opt.clone().cpu(), torch.full(size= (z.shape[0], 1), fill_value= float('inf'))

    patient = 0
    for _ in range(max_iter):
        pred = pmodel(z_opt)
        loss = criterion(pred, target)
        optimizer.zero_grad()
        pmodel.zero_grad()
        loss.mean().backward()
        optimizer.step()

        loss = loss.mean(dim= -1, keepdim= True).cpu()
        if torch.all(loss>=best_loss):
            patient += 1
        else:
            best_z = torch.where(loss<best_loss, z_opt.clone().cpu(), best_z).detach()
            best_loss = torch.where(loss<best_loss, loss, best_loss).detach()

        if patient > max_patient:
            break

    return best_z


def constrained_prop_optimize(mols: list,
                              target: torch.Tensor,
                              scoring_func,              # scoring_func(mol) -> score
                              model: nn.Module,
                              pmodel: nn.Module,
                              labels: list,
                              frag_ecfps: torch.Tensor,
                              ndummys: torch.Tensor,
                              min_size: int= 1,
                              max_nfrags: int= 30, 
                              useChiral: bool= True,
                              ignore_double: bool= False,
                              criterion: nn.Module= nn.MSELoss(reduction= 'none'),
                              thresholds: list= [0.2, 0.4, 0.6],
                              lr: float= 0.01,
                              max_iter: int= 80,
                              device: torch.device= torch.device('cpu')
                              ) -> list:
    """
    Multiple properties not supported.
    Pmodel outputs have shape= (batch_size, 1)
    """
    z_all = []
    for mol in mols:
        frags, bond_types, bondMapNums, _ = debugMolToBRICSfragments(mol, minFragSize= min_size, maxFragNums= max_nfrags, 
                                                                     maxDegree= model.width, useChiral= useChiral, ignore_double= ignore_double)
        frag_idxs = [labels.index(f) for f in frags]
        frag_idxs, features, positions = get_tree_features(frag_idxs, frag_ecfps[frag_idxs], bond_types, bondMapNums, model.depth, model.width, False)
        frag_idxs, features, positions = frag_idxs.unsqueeze(0), features.unsqueeze(0), positions.unsqueeze(0)
        src = torch.hstack([torch.full((frag_idxs.shape[0], 1), -1), frag_idxs.detach()])
        src_mask, _, src_pad_mask, _ = create_mask(src, src, pad_idx= 0, batch_first= True)
        z, _, _ = model.encode(features.to(device), positions.to(device), src_mask.to(device), src_pad_mask.to(device))
        z_all.append(z.cpu())
    z_all = torch.vstack(z_all)

    model.set_labels(labels)
    smiles_opt = []
    z_opt = z_all.clone()
    for _ in range(max_iter):
        z_opt = prop_optimize(z_opt, target, pmodel, criterion, lr= lr, max_iter= 1, device= device)
        with torch.no_grad():
            smis_opt = model.sequential_decode(z_opt.to(device), frag_ecfps, ndummys, max_nfrags= max_nfrags, asSmiles= True)
            smiles_opt.append(smis_opt)

    improved_smiles, improved_score = {f'thr-{thr}': [] for thr in thresholds}, {f'thr-{thr}': [] for thr in thresholds}
    for mol, smis_opt in zip(mols, zip(*smiles_opt)):
        ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        ecfps_opt = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048) for s in smis_opt]
        tanimotos = DataStructs.BulkTanimotoSimilarity(ecfp, ecfps_opt)
        tanimotos = np.where(tanimotos==1, 0, tanimotos)    # remove the same mols

        orig_score = scoring_func(mol)
        for thr in thresholds:
            smis_cand = [smis_opt[i] for i in range(max_iter) if tanimotos[i] >= thr]
            scores = np.array([scoring_func(Chem.MolFromSmiles(s)) for s in smis_cand])

            if np.any(scores>orig_score):
                max_improve = scores.max() - orig_score
                improved_smi = smis_cand[np.argmax(scores)]
            else:
                max_improve = float('nan')
                improved_smi = None
            improved_smiles[f'thr-{thr}'].append(improved_smi)
            improved_score[f'thr-{thr}'].append(max_improve)
    
    return improved_smiles, improved_score


def featurelize_fragments(dataloader,
                          model: nn.Module,
                          frag_ecfps: torch.Tensor,
                          root: bool= False,
                          device: torch.device= torch.device('cpu')
                          ):
    attn_dict = {i: [] for i in range(len(frag_ecfps))}
    start = time.time()
    with torch.no_grad():
        for iter, data in enumerate(dataloader):
            frag_indices = data[0]
            features = frag_ecfps[frag_indices.flatten()].reshape(frag_indices.shape[0], frag_indices.shape[1], -1).to(device)
            positions = data[1].to(device)

            # make mask
            src = torch.hstack([torch.full((frag_indices.shape[0], 1), -1), frag_indices.detach()]).to(device)  # for super root
            src_mask, _, src_pad_mask, _ = create_mask(src, src, pad_idx= 0, batch_first= True)

            # positional embbeding
            src = model.fc_ecfp(features) + model.PE(positions)           # (B, L, d_model)

            # attach super root
            root_embed = model.embed(src.new_zeros(src.shape[0], 1).long())     
            src = torch.cat([root_embed, src], dim= 1)                    # (B, L+1, d_model)

            x = src
            attn_values = []
            for layer in model.encoder.layers:
                attn, attn_map = layer.self_attn(x, x, x, attn_mask= src_mask, key_padding_mask= src_pad_mask, need_weights= True)
                x = layer(x, src_mask, src_pad_mask)
                attn_value = attn_map[:, 0, :] if root else attn_map.mean(dim= 1)  # if True, use only root node values
                attn_values.append(attn_value.unsqueeze(-1))
            attn_values = torch.stack(attn_values, dim= -1)                  # (B, L+1, len(layers))
        
            for idxs, attn in zip(frag_indices, attn_values):
                for i, values in enumerate(attn[1:]):
                    attn_dict[idxs[i].item()].append(values.squeeze(0).tolist())

            if (iter==0) | (((iter+1) % 10) == 0) | (iter==len(dataloader)-1):
                print(f'[{iter+1}/{len(dataloader)}] elapsed time: {second2date(time.time()-start)}', flush= True)
                start = time.time()
    
    attn_mean, attn_std = [], []
    for key, values in attn_dict.items():
        values = np.array(values)
        if len(values) == 0:
            mu = [0] * len(model.encoder.layers)
            std = [0] * len(model.encoder.layers)
        else:
            mu = values.mean(axis= 0).tolist()
            std = values.std(axis= 0).tolist()
        attn_mean.append(mu)
        attn_std.append(std) 
    del attn_dict

    return attn_mean, attn_std
            

if __name__ == '__main__':
    pmodel = nn.Linear(128, 2)
    z = torch.rand(6, 128)
    target = torch.ones(6, 2)

    best_z = prop_optimize(z, target, pmodel)
    print(best_z)