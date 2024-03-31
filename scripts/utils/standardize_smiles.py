import argparse
import copy
import random
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

import moses
from molvs import  Standardizer


parser = argparse.ArgumentParser(description= 'please enter paths')
parser.add_argument('data_path', type= str, help= 'csv file')
parser.add_argument('--keep_dupls', action= 'store_true')
parser.add_argument('--n_jobs', type= int, default= 24, help= 'the number of cpu for parallel, default 24')
args = parser.parse_args()

data_path = args.data_path
df = pd.read_csv(data_path)
if 'SMILES' not in df.columns:
    raise ValueError('Please change the name of column smiles to "SMILES"')
fname = data_path.rsplit('.csv')[0]
print(f'loaded: {args.data_path}', flush= True)

stand = Standardizer()

def clearAtomMapNums(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

def standardize_and_metrics(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        mol = copy.deepcopy(mol)
        clearAtomMapNums(mol)
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
        # mol = stand.disconnect_metals(mol)
        mol = stand.normalize(mol)
        mol = stand.reionize(mol)
        # Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        smi_std = Chem.MolToSmiles(mol)

        natom = mol.GetNumAtoms()
        mw = moses.metrics.weight(mol)
        logP = moses.metrics.logP(mol)
        qed = moses.metrics.QED(mol)
        sa = moses.metrics.SA(mol)
        npl = moses.metrics.NP(mol)
        tpsa = Chem.Descriptors.TPSA(mol)
        ct = Descriptors.BertzCT(mol)
    except:
        mol = smi_std = None
        natom = mw = logP = qed = sa = npl = tpsa = ct =  None

    if smi_std is None:
        print('none', flush= True)
    
    return smi, smi_std, natom, mw, logP, qed, sa, npl, tpsa, ct

# standardize
results = Parallel(n_jobs= args.n_jobs)(delayed(standardize_and_metrics)(s) for s in df.SMILES)
smiles, smiles_std, natoms_list, mw_list, logP_list, qed_list, sa_list, npl_list, tpsa_list, ct_list = map(list, zip(*results))

# train-test split
assert len(df) == len(smiles_std)
try:
    test = df.test.tolist()
except:
    random.seed(0)
    test = random.choices([0, 1, -1], k= len(df), weights= [0.90, 0.05, 0.05])
    print('random train-valid-test split. train:valid:test= 0.90:0.05:0.05', flush= True)

# add columns
df['SMILES'] = smiles_std
df['test'] = test
df_tmp = pd.DataFrame({'natoms': natoms_list, 'MW': mw_list, 'logP': logP_list, 'QED': qed_list, 
                       'SA': sa_list, 'NP': npl_list, 'TPSA': tpsa_list, 'BertzCT': ct_list})
df_new = pd.concat([df, df_tmp], axis= 1)

# remove errors
df_error = df_new.loc[df_new['SMILES'].isnull()]
n_errors = len(df_error)
df_new = df_new.loc[~df_new['SMILES'].isnull()].reset_index(drop= True)

# check duplications
dupls = df_new.duplicated(subset= 'SMILES', keep= 'first')
n_dupls = sum(dupls)
if not args.keep_dupls:
    df_error = pd.concat([df_error, df_new.loc[dupls]])
    df_new = df_new.loc[~dupls].reset_index(drop= True)

# save
df_new.to_csv(f'{fname}_standardized.csv', index= False)
if len(df_error) > 0:
    df_error.to_csv(f'{fname}_removed.csv', index= False)

# output
with open(f'{fname}_stats.txt', 'a') as f:
    for col in df_tmp.columns:
        f.write(f'-{col}: {np.nanmean(df_tmp[col]):.4f}±{np.nanstd(df_tmp[col]):.4f} ({np.nanmin(df_tmp[col]):.4f}-{np.nanmax(df_tmp[col]):.4f})')

print(f'[BEFORE] {len(df)}', flush= True)
print(f'[AFTER] {len(df_new)}', flush= True)
print(f'removed:  {len(df_error)} (error: {n_errors}, dupl: {n_dupls}, keep_dupls: {args.keep_dupls})', flush= True)
