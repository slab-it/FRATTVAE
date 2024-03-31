import os
import sys
import gc
import time
from typing import List
from functools import partial
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import pickle
import yaml
from rdkit import Chem
import timeout_decorator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mso.optimizer import BasePSOptimizer, ParallelSwarmOptimizer
from mso.objectives.scoring import ScoringFunction
from guacamol.goal_directed_generator import GoalDirectedGenerator

sys.path.append(os.path.abspath('../../scripts'))
from models.fttvae import FTTVAE
from utils.preprocess import debugMolToBRICSfragments, SmilesToMorganFingetPrints, frag2ecfp
from utils.tree import make_tree, get_pad_features
from utils.mask import create_mask

# yml_file = '/yourdirectory/fttvae/results/struct_useChiral1_uesugi_max32_1103/input_data/params.yml'


def smiles2mol(smiles, decoy= 'c1ccccc1'):
    if smiles is None:
        print(f'[CAUTION] NoneType to {decoy}', flush= True)
        return Chem.MolFromSmiles(decoy)
    else:
        return Chem.MolFromSmiles(smiles)

def smiles2tree(smiles, n_bits, radius, depth, width, 
                max_nfrags, useChiral, ignore_double, ignore_dummy):
    mol = smiles2mol(smiles)
    frags, bond_types, bondMapNums, _ = debugMolToBRICSfragments(mol, minFragSize= 1, maxFragNums= max_nfrags, 
                                                                 maxDegree= width, useChiral= useChiral, ignore_double= ignore_double)
    ecfps = torch.tensor([frag2ecfp(Chem.MolFromSmiles(f), n_bits= n_bits, radius= radius, useChiral= useChiral, ignore_dummy= ignore_dummy) for f in frags]).float()
    decoy_idxs = list(range(1, len(frags)+1))
    tree = make_tree(decoy_idxs, ecfps, bond_types, bondMapNums)
    tree.set_all_positional_encoding(d_pos= depth * width, n= width)
    
    return tree


class InferenceModel(nn.Module):
    def __init__(self, yml_file: str, load_epoch: int= None, 
                 device: torch.device= 'cpu', n_jobs: int= 1, max_length: int= 2000) -> None:
        super().__init__()
        self.yml_file = yml_file
        self.n_jobs = n_jobs
        self.max_length = max_length
        self.device = device
        with open(yml_file) as yml:
            params = yaml.safe_load(yml)

        model_params = params['model']
        decomp_params = params['decomp']
        self.n_bits = decomp_params['n_bits']
        self.dupl_bits = decomp_params['dupl_bits']
        self.max_nfrags = decomp_params['max_nfrags']
        self.radius = decomp_params['radius']
        self.max_depth = decomp_params['max_depth']
        self.max_degree = decomp_params['max_degree']
        self.useChiral = decomp_params['useChiral']
        self.ignore_double = decomp_params['ignore_double']
        self.ignore_dummy = decomp_params['ignore_dummy']
        self.batch_size = params['train']['batch_size']

        df_frag = pd.read_csv(params['frag_path'])
        uni_fragments = df_frag['SMILES'].tolist()
        try:
            with open(os.path.join(params['result_path'], 'input_data', 'csr_ecfps.pkl'), 'rb') as f:
                frag_ecfps = pickle.load(f).toarray()
                self.frag_ecfps = torch.from_numpy(frag_ecfps).float()
            assert self.frag_ecfps.shape[0] == len(uni_fragments)
            assert self.frag_ecfps.shape[1] == (self.n_bits + self.dupl_bits)
        except Exception as e:
            print(e, flush= True)
            frag_ecfps = torch.tensor(SmilesToMorganFingetPrints(uni_fragments[1:], n_bits= self.n_bits, dupl_bits= 0, radius= self.radius, 
                                                                ignore_dummy= self.ignore_dummy, useChiral= self.useChiral, n_jobs= n_jobs)).float()
            self.frag_ecfps = torch.vstack([frag_ecfps.new_zeros(1, self.n_bits), frag_ecfps])      # padding feature is zero vector
        self.ndummys = torch.tensor(df_frag['ndummys'].tolist()).long()

        num_labels = self.frag_ecfps.shape[0]
        self.model = FTTVAE(num_labels, self.max_depth, self.max_degree, 
                            model_params['feat'], model_params['latent'], 
                            model_params['d_model'], model_params['d_ff'], model_params['nlayer'], 
                            model_params['nhead'], model_params['activation']).to(self.device)
        if load_epoch:
            load_epoch = load_epoch
            self.model.load_state_dict(torch.load(os.path.join(params['result_path'], 'models', f'model_iter{load_epoch}.pth'), map_location= device))
        else:
            load_epoch = '_best'
            self.model.load_state_dict(torch.load(os.path.join(params['result_path'], 'models', f'model_best.pth'), map_location= device))
        self.model.set_labels(uni_fragments)
        self.model.n_jobs = n_jobs
        self.model.eval()

        del df_frag, uni_fragments, frag_ecfps
        gc.collect()

    def seq_to_emb(self, smiles: list):
        if isinstance(smiles, str):
            smiles = [smiles]
        tree_list = Parallel(n_jobs= min(len(smiles), self.n_jobs))(
            delayed(smiles2tree)(s, self.n_bits, self.radius, self.max_depth, self.max_degree, 
                                 self.max_nfrags, self.useChiral, self.ignore_double, self.ignore_dummy) for s in smiles)
        indices = get_pad_features(tree_list, key= 'fid', max_nodes_num= self.max_nfrags).squeeze(-1)
        features = get_pad_features(tree_list, key= 'x', max_nodes_num= self.max_nfrags)
        positions = get_pad_features(tree_list, key= 'pos', max_nodes_num= self.max_nfrags)

        loader = DataLoader(TensorDataset(indices, features, positions), shuffle= False, batch_size= self.batch_size)
        z_all = []
        with torch.no_grad():
            for data in loader:
                indices, features, positions = data[0], data[1], data[2]
                src = torch.hstack([torch.full((indices.shape[0], 1), -1), indices.detach()]).to(self.device)  # for super root
                src_mask, _, src_pad_mask, _ = create_mask(src, src, pad_idx= 0, batch_first= True)
                z, _, _ = self.model.encode(features.to(self.device), positions.to(self.device), src_mask, src_pad_mask)
                z_all.append(z.cpu())
        return torch.vstack(z_all).numpy()

    def emb_to_seq(self, z: np.ndarray):
        smiles = []
        loader = DataLoader(TensorDataset(torch.from_numpy(z).float()), shuffle= False, batch_size= self.batch_size)
        with torch.no_grad():
            for z in loader:
                smiles += self.model.sequential_decode(z[0].to(self.device), self.frag_ecfps, self.ndummys, max_nfrags= self.max_nfrags, asSmiles= True)
        smiles = [s if (s is not None) & (len(s)<self.max_length) else 'c1ccccc1' for s in smiles]
        return smiles
    
    def random_generate(self, n):
        smiles = []
        loader = DataLoader(TensorDataset(torch.randn(size= (n, self.model.latent_dim))), shuffle= False, batch_size= self.batch_size)
        with torch.no_grad():
            for z in loader:
                smiles += self.model.sequential_decode(z[0].to(self.device), self.frag_ecfps, self.ndummys, max_nfrags= self.max_nfrags, asSmiles= True)
        return smiles

class GuacaGoalInference(GoalDirectedGenerator):
    def __init__(self, model: InferenceModel, config: dict= None) -> None:
        super().__init__()
        self.model = model
        self.ntask = 0
        self.config = {'restart': 40, 'iter': 250, 'n_swarms': 1, 'n_part': 200, 'x_max': 1.0, 'x_min': -1.0} if config is None else config
 
    def generate_optimized_molecules(self, 
                                     scoring_function, 
                                     number_molecules: int, 
                                     starting_population: List[str] = None) -> List[str]:
        start = time.time()
        self.ntask += 1
        desire = [{"x": -2.0, "y": -2.0}, {"x": 1.0, "y": 1.0}]
        scoring_func_from_mol = partial(self._scoring_from_mol, scoring_func= scoring_function.score)
        scoring_functions = [ScoringFunction(scoring_func_from_mol, name= 'none', is_mol_func= True, desirability= desire)]

        best_solutions = pd.DataFrame(columns=["smiles", "fitness"])
        for i in range(self.config['restart']):
            init_smiles = self.model.random_generate(self.config['n_swarms']) if starting_population is None else starting_population
            # init_smiles = ['c1ccccc1'] if starting_population is None else starting_population
            print('Init Smiles:', init_smiles, flush= True)
            if self.config['n_swarms'] > 1:
                opt = ParallelSwarmOptimizer.from_query_list(init_smiles= init_smiles,
                                                            num_part= self.config['n_part'],
                                                            num_swarms= len(init_smiles),
                                                            inference_model= self.model,
                                                            scoring_functions= scoring_functions,
                                                            x_max= self.config['x_max'],
                                                            x_min= self.config['x_min']
                                                            )
            else:
                opt = ParallelSwarmOptimizer.from_query(init_smiles= init_smiles,
                                                        num_part= self.config['n_part'],
                                                        num_swarms= self.config['n_swarms'],
                                                        inference_model= self.model,
                                                        scoring_functions= scoring_functions,
                                                        x_max= self.config['x_max'],
                                                        x_min= self.config['x_min']
                                                        )
            opt.run(self.config['iter'], num_track= 250)
            opt.best_solutions.to_csv(f'best_solutions{self.ntask}.csv', index= False, mode= 'a')
            opt_smiles = sorted(opt.best_solutions['smiles'].tolist(), key= lambda s: scoring_function.score(s), reverse= True)
            print(f'[{i+1}/{self.config["restart"]}] Optimized Smiles: {opt_smiles[0]} ({scoring_function.score(opt_smiles[0]):.4f}), elapsed time: {int(time.time()-start)} s', flush= True)
            best_solutions = pd.concat([best_solutions, opt.best_solutions])

        opt_smiles = sorted(best_solutions['smiles'].drop_duplicates().tolist(), key= lambda s: scoring_function.score(s), reverse= True)
        print(f'[Task-{self.ntask}] Optimized Smiles: {opt_smiles[0]} ({scoring_function.score(opt_smiles[0]):.4f}), elapsed time: {int(time.time()-start)} s\n', flush= True)
        return opt_smiles[:number_molecules]
    
    @staticmethod
    def _scoring_from_mol(mol, scoring_func, decoy_value: float= 0.5, timeout: int= 3600):
        smi = Chem.MolToSmiles(mol)
        @timeout_decorator.timeout(timeout, use_signals= False)
        def scoring_func_with_timeout(smi):
            return scoring_func(smi)
        
        try:
            return scoring_func_with_timeout(smi)
        except:
            print(f'{smi} is timeout', flush= True)
            return decoy_value