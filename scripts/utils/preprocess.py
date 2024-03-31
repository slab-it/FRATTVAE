import sys
from collections import Counter
from copy import deepcopy
from itertools import chain
import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from molvs import standardize_smiles
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from tqdm import tqdm

from utils.decompose import MolFromFragments, MolToBRICSfragments, MapNumsToAdj

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def smiles2mol(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        print(f'[ERROR] {s} is not valid.', flush= True)
    return m

def frag2ecfp(frag, radius: int= 2, n_bits: int= 2048, useChiral: bool= True, ignore_dummy: bool= False):
    if ignore_dummy:
        frag = Chem.RemoveHs(AllChem.ReplaceSubstructs(frag, Chem.MolFromSmiles('*'), Chem.MolFromSmiles('[H]'), replaceAll= True)[0])
    return AllChem.GetMorganFingerprintAsBitVect(frag, radius, n_bits, useChirality= useChiral)

def FragmentsToIndices(fragments_list: list, fragment_idxs: dict, verbose: int= 0):
    if verbose:
        return [[fragment_idxs[f] for f in frags] for frags in tqdm(fragments_list)]
    else:
        return [[fragment_idxs[f] for f in frags] for frags in fragments_list]


def MolsToBRICSfragments(mols: list, debug: bool= True, 
                         useChiral: bool= True, ignore_double: bool= False, 
                         minFragSize: int= 1, maxFragNums: int= 50, maxDegree: int= 32,
                         uni_fragments: list= None, asFragments: bool= False,):
    fragments_list = []
    bondtypes_list = []
    bondMapNums_list = []
    all_fragments = []
    recon_flag = []

    # decompose mol
    for i, mol in enumerate(mols):
        s = Chem.MolToSmiles(mol)
        # hasChiral = bool(Chem.FindMolChiralCenters(mol)) if useChiral else False
        hasChiral = bool(len(Chem.FindPotentialStereo(mol, flagPossible= False))) if useChiral else False
        frags, bond_types, bondMapNums = MolToBRICSfragments(mol, minFragSize= minFragSize, maxDegree= maxDegree, useChiral= useChiral, useStereo= ignore_double)
        min_size = minFragSize
        while len(frags) > maxFragNums:
            min_size += 1
            frags, bond_types, bondMapNums = MolToBRICSfragments(mol, minFragSize= minFragSize, maxDegree= maxDegree, useChiral= hasChiral, useStereo= ignore_double)
            print(f"{s} has over {maxFragNums} fragments. min_size:{min_size-1} -> {min_size}", flush= True)
        if debug:
            try:
                adj = MapNumsToAdj(bondMapNums, bond_types)
                s1 = Chem.MolToSmiles(mol, isomericSmiles= hasChiral)
                s2 = Chem.MolToSmiles(MolFromFragments(frags, adj, asMol= True), isomericSmiles= hasChiral)
                if (s1 != s2) & (s1 != standardize_smiles(s2)):
                    if hasChiral:
                        s1_dash = Chem.CanonSmiles(s1, useChiral= 0)
                        s2_dash = Chem.CanonSmiles(s2, useChiral= 0)
                        if (s1_dash == s2_dash) | (s1_dash == standardize_smiles(s2_dash)):
                            recon = 2
                            print(f"[{i}] {s1}, {s2} is 3D unreconstructable.", flush= True)
                        else:
                            recon = 0
                            print(f"[{i}] {s1_dash}, {s2_dash} is 2D unreconstructable.", flush= True)
                    else:
                        recon = 0
                        print(f"[{i}] {s1}, {s2} is unreconstructable.", flush= True)
                else:
                    recon = 3 if hasChiral else 1
            except:
                recon = 0
                print(f'[{i}] {s} is an ERROR.')

        if recon > 0:
            all_fragments += frags
            fragments_list.append(frags)
            bondtypes_list.append(bond_types)
            bondMapNums_list.append(bondMapNums)
            recon_flag.append(recon)

    # calculate frequency and Uniqueness
    if uni_fragments is None:
        frag_freq = Counter(all_fragments)
        uni_fragments, freq_list = map(list, zip(*frag_freq.items()))
        df = pd.DataFrame({'SMILES': uni_fragments, 'frequency': freq_list})
        df = df.assign(length= df.SMILES.str.len())
        df = df.sort_values(['frequency', 'length'], ascending= [False, True])
        uni_fragments = df.SMILES.tolist()
        freq_list = df.frequency.tolist()
        uni_fragments = ['*'] + uni_fragments
        fragments_list = [0] + freq_list
    else:
        freq_list = None
    
    if not asFragments:
        fragment_idxs = dict(zip(uni_fragments, range(len(uni_fragments))))
        fragments_list = FragmentsToIndices(fragments_list, fragment_idxs)

    return fragments_list, bondtypes_list, bondMapNums_list, recon_flag, uni_fragments, freq_list
    

def debugMolToBRICSfragments(mol, 
                             useChiral: bool= True, ignore_double: bool= False, 
                             minFragSize: int= 1, maxFragNums: int= 50, maxDegree: int= 32):
    recon = 1
    iters = 0
    max_iters = 30
    try:
        # decompose
        s = Chem.MolToSmiles(mol) if mol is not None else None
        Chem.FindMolChiralCenters(mol)  # assign chirality
        hasChiral = bool(len(Chem.FindPotentialStereo(mol, flagPossible= False))) if useChiral else False
        frags, bond_types, bondMapNums = MolToBRICSfragments(mol, minFragSize= minFragSize, maxDegree= maxDegree, useChiral= hasChiral, useStereo= ignore_double)
        while (len(frags) > maxFragNums):
            iters += 1
            minFragSize += 1
            frags, bond_types, bondMapNums = MolToBRICSfragments(mol, minFragSize= minFragSize, maxDegree= maxDegree, useChiral= hasChiral, useStereo= ignore_double)
            # print(f"{s} has over {maxFragNums} fragments. min_size:{minFragSize-1} -> {minFragSize}", flush= True
            if iters > max_iters:
                raise ValueError(f'Over max iteration; {max_iters}. Remove it or increse max_nfrags.')

        # reconstruct    
        adj = MapNumsToAdj(bondMapNums, bond_types)
        s1 = Chem.MolToSmiles(mol, isomericSmiles= hasChiral)
        s2 = Chem.MolToSmiles(MolFromFragments(frags, adj, asMol= True), isomericSmiles= hasChiral)
        if (s1 != s2) & (s1 != standardize_smiles(s2)):
            if hasChiral:
                s1_dash = Chem.CanonSmiles(s1, useChiral= 0)
                s2_dash = Chem.CanonSmiles(s2, useChiral= 0)
                if (s1_dash == s2_dash) | (s1_dash == standardize_smiles(s2_dash)):
                    recon = 2
                    # print(f"{s1}, {s2} is 3D unreconstructable.", flush= True)
                else:
                    recon = 0
                    print(f"{s1_dash}, {s2_dash} is 2D unreconstructable.", flush= True)
            else:
                recon = 0
                print(f"{s}, {s2} is unreconstructable.", flush= True)
        else:
            recon = 3 if hasChiral else 1
    except Exception as e:
        recon = 0
        print(f'{s} is an ERROR; {str(e)}', flush= True)
    
    if recon == 0:
        frags, bond_types, bondMapNums = None, None, None

    return frags, bond_types, bondMapNums, recon


def parallelMolsToBRICSfragments(mols: list,
                                 useChiral: bool= True, ignore_double: bool= False,
                                 minFragSize: int= 1, maxFragNums: int= 50, maxDegree: int= 32,
                                 df_frag: pd.DataFrame= {}, asFragments: bool= False,
                                 n_jobs: int= -1, verbose: int= 0):
    # decompose mol
    results = Parallel(n_jobs= n_jobs, verbose= verbose)(
        delayed(debugMolToBRICSfragments)(mol, minFragSize= minFragSize, maxFragNums= maxFragNums, maxDegree= maxDegree, useChiral= useChiral, ignore_double= ignore_double) for mol in mols)
    fragments_list, bondtypes_list, bondMapNums_list, recon_flag = zip(*results)
    fragments_list, bondtypes_list, bondMapNums_list, recon_flag = list(fragments_list), list(bondtypes_list), list(bondMapNums_list), list(recon_flag)

    # remove None
    for _ in range(fragments_list.count(None)):
        fragments_list.remove(None)
        bondtypes_list.remove(None)
        bondMapNums_list.remove(None)

    # calculate frequency and Uniqueness
    all_fragments = list(chain.from_iterable(fragments_list))
    frag_freq = Counter(all_fragments)
    uni_fragments, freq_list = map(list, zip(*frag_freq.items()))
    df = pd.DataFrame({'SMILES': uni_fragments, 'frequency': freq_list})
    df = df.assign(length= df.SMILES.str.len())
    df = df.sort_values(['frequency', 'length'], ascending= [False, True])

    # merge origin fragments list
    if len(df_frag):
        df = pd.concat([df_frag, df]).drop_duplicates(subset= 'SMILES', keep= 'first').reset_index(drop= True)
        uni_fragments = df.SMILES.tolist()
        freq_list = df.frequency.tolist()
    else:
        uni_fragments = df.SMILES.tolist()
        freq_list = df.frequency.tolist()
        uni_fragments = ['*'] + uni_fragments
        freq_list = [0] + freq_list

    # index
    if not asFragments:
        fragment_idxs = dict(zip(uni_fragments, range(len(uni_fragments))))
        fragments_list = FragmentsToIndices(fragments_list, fragment_idxs, verbose= verbose)

    return fragments_list, bondtypes_list, bondMapNums_list, recon_flag, uni_fragments, freq_list


def _smilesToMorganFingarPrintsAsBitVect(smiles, radius: int, n_bits: int, useChiral: bool= True, ignore_dummy: bool= True):
    ecfp = frag2ecfp(Chem.MolFromSmiles(smiles), radius, n_bits, useChiral= useChiral, ignore_dummy= ignore_dummy)
    ecfp_bits = np.array(ecfp, dtype= int).tolist()

    return ecfp, ecfp_bits


def SmilesToMorganFingetPrints(fragments: list, n_bits: int, dupl_bits: int= 0, radius: int= 2, 
                               ignore_dummy: bool= True, useChiral: bool= True, n_jobs: int= 20):
    """
    fragments: a list of fragment smiles.
    distinct_ecfp: If fragments list includes same ecfp, add bits to distinguish ecfp.
    """
    results = Parallel(n_jobs= n_jobs)(delayed(_smilesToMorganFingarPrintsAsBitVect)(f, radius, n_bits, useChiral= useChiral, ignore_dummy= ignore_dummy) for f in fragments)
    ecfps, ecfp_list = map(list, zip(*results))

    if dupl_bits > 0:
        _, indices = np.unique(ecfp_list, axis= 0, return_index= True)
        uni_ecfps = [ecfps[i] for i in indices]
        dupl_idxs = []
        for e in uni_ecfps:
            dupl = np.where(np.array(DataStructs.BulkTanimotoSimilarity(e, ecfps)) == 1)[0]
            if len(dupl) >= 2:
                dupl_idxs.append(sorted(dupl.tolist()))

        if len(dupl_idxs) != 0:
            # add bits to ecfp
            max_len = max(list(map(len, dupl_idxs)))
            if max_len > 2**dupl_bits:
                raise ValueError(f'the number of duplicated ecfp {max_len} is greater than 2**{dupl_bits}')
            bits = [list(map(lambda x: int(x), list(f"{i:0{dupl_bits}b}"))) for i in range(max_len)]
            distinct_ecfp_list = [ecfp_list + bits[0] for ecfp_list in ecfp_list]
            for dupl in dupl_idxs:
                for i, d in enumerate(dupl):
                    distinct_ecfp_list[d] = distinct_ecfp_list[d][:-dupl_bits] + bits[i]
            ecfp_list = distinct_ecfp_list
        print(f'the number of duplicated ecfp is {len(dupl_idxs)}', flush= True)
        
    return ecfp_list


def IndicesToFeatures(indices_mols: list, features: list):
    mol_features_list = []
    for indices in indices_mols:
        mol_features_list.append([features[i] for i in indices])

    return mol_features_list


if __name__ == '__main__':
    smi = '[2H]C([2H])([2H])c1ccnc2c1NC(=O)c1cccnc1N2C1CC1'
    frag, bond, maps, recon = debugMolToBRICSfragments(Chem.MolFromSmiles(smi), ignore_double= False)
    adj = MapNumsToAdj(maps, bond)
    smi_rev = MolFromFragments(frag, adj)
    print(smi)
    print(smi_rev)