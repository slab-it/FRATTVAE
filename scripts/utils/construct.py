import numpy as np
from copy import deepcopy
from itertools import chain

import timeout_decorator

from rdkit import Chem, rdBase, RDLogger
from rdkit.Chem import RWMol, AllChem, DataStructs
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
# from molvs import Standardizer

BONDTYPES = {1: Chem.rdchem.BondType.SINGLE, 
             2: Chem.rdchem.BondType.DOUBLE,
             3: Chem.rdchem.BondType.TRIPLE}


def isomer_search(smiles, ecfp: np.ndarray, radius: int= 2):
    if type(ecfp) == list:
        ecfp = np.array(ecfp)
    if smiles is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    n_bits = len(ecfp)
    opts = StereoEnumerationOptions(unique= True, onlyUnassigned= True)
    isomers = list(EnumerateStereoisomers(mol, options=opts))
    if len(isomers) == 0:
        return smiles
    else:
        isomers += [mol]
    euclids = []
    for isomer in isomers:
        iso_ecfp = np.array(AllChem.GetMorganFingerprintAsBitVect(isomer, radius, n_bits, useChirality= True))
        euclids.append(np.linalg.norm(ecfp-iso_ecfp))
        
    return Chem.MolToSmiles(isomers[np.argmin(euclids)])


def calc_tanimoto(smi1, smi2, useChiral: bool= True):
    if (smi1 is None) | (smi2 is None):
        return float('nan')
    ecfp1 = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi1), 2, 2048, useChirality= useChiral)
    ecfp2 = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi2), 2, 2048, useChirality= useChiral)
    return DataStructs.TanimotoSimilarity(ecfp1, ecfp2)


def reconstructMol(smiles, frags, adj, ecfp: list= None,
                   useChiral: bool= True, verbose: bool= False):
    # hasChiral = bool(Chem.FindMolChiralCenters(Chem.MolFromSmiles(smiles))) if useChiral else False
    hasChiral = bool(len(Chem.FindPotentialStereo(Chem.MolFromSmiles(smiles), flagPossible= False))) if useChiral else False
    try:
        # rev_smiles = MolFromFragments(frags, adj, asMol= False, useChiral= hasChiral)
        rev_smiles = constructMol(frags, adj, asMol= False, useChiral= hasChiral)
        if rev_smiles is None:
            raise ValueError()
        
        if hasChiral & (ecfp is not None):
            rev_smiles = isomer_search(rev_smiles, ecfp)

        if Chem.CanonSmiles(smiles, useChiral= int(hasChiral)) != Chem.CanonSmiles(rev_smiles, useChiral= int(hasChiral)):
            if Chem.CanonSmiles(smiles, useChiral= 0) == Chem.CanonSmiles(rev_smiles, useChiral= 0):
                correct = 2
            else:
                correct = 0
                if verbose: print(f'[UNMATCH] {smiles}, {rev_smiles}', flush= True)
        else:
            correct = 3 if hasChiral else 1
    except:
        rev_smiles = None
        correct = 0
        if verbose: print(f'[ERROR] {smiles}', flush= True)

    return rev_smiles, correct


def constructMol(frag_smiles: list, adj: list, asMol: bool= False, useChiral: bool= True):
    """ 
        frag_smiles: list of fragments as SMILES,
        adj: torch.Tensor or list shape= (len(frag_smiles), len(frag_smiles)), 0: none, 1: single, 2: double, 3: triple
        validity correction: choose largest fragments and check Valence
    """
    # process options
    blg = rdBase.BlockLogs()
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)
    # stand = Standardizer()
    remover = Chem.RemoveHsParameters()
    remover.removeDegreeZero = False

    # one node
    if len(adj) < 2:
        mol = AllChem.ReplaceSubstructs(Chem.MolFromSmiles(frag_smiles[0]), Chem.MolFromSmiles('*'), Chem.MolFromSmiles('[H]'), replaceAll= True)[0]
        mol = Chem.RemoveHs(mol, remover, sanitize= False)
        # mol = stand.standardize(mol)
        smi = Chem.MolToSmiles(mol)
        try:
            if Chem.SanitizeMol(mol, catchErrors= True) == Chem.rdmolops.SanitizeFlags.SANITIZE_KEKULIZE:
                # print(smi, flush= True)
                smi = smi.replace('[n+]', '[n]')
            smi = Chem.CanonSmiles(smi)
            if asMol: smi = Chem.MolFromSmiles(smi)
        except:
            smi = None
        return smi
    else:
        adj = np.array(adj)

    # smiles to mol
    fragments = [Chem.MolFromSmiles(s) for s in frag_smiles]

    # assign AtomMapNum and combine fragments
    for i, frag in enumerate(fragments):
        n = delta = 0.25
        for atom in frag.GetAtoms():
            if atom.GetAtomicNum() == 0:
                if atom.GetAtomMapNum() == 0:
                    assert n < 10
                    atom.SetDoubleProp('dummyMapNumber', n + 10*i)
                    n += delta
                else:
                    atom.SetDoubleProp('dummyMapNumber', delta * atom.GetAtomMapNum() + 10*i)
        if i == 0:
            combo = frag
        else:
            combo = Chem.CombineMols(combo, frag)
    rwcombo = RWMol(combo)

    tril = np.tril(adj)
    indices = list(map(list, np.where(tril)))
    bond_types = tril[tril!=0].tolist()
    stereos = [list(bond.GetStereoAtoms()) for bond in rwcombo.GetBonds() if bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE]
    stereo_atoms = list(chain.from_iterable(stereos))

    # add Bond between fragments
    for i1, i2, b in zip(*indices, bond_types):
        start_end = [i1, i2]
        connect_idxs = []
        dummy_idxs = []
        bonds = []
        for idx in start_end:
            for i, a in enumerate(rwcombo.GetAtoms()):
                if a.HasProp('dummyMapNumber'):
                    mapnum = a.GetDoubleProp('dummyMapNumber')
                    if mapnum == 10*idx + delta:
                        a.ClearProp('dummyMapNumber')
                        if len(a.GetNeighbors()) == 1:
                            neighbor = a.GetNeighbors()[0]
                            dummy_idxs.append(a.GetIdx())
                            connect_idxs.append(neighbor.GetIdx())
                            bonds.append(int(rwcombo.GetBondBetweenAtoms(a.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble()))
                    elif (mapnum > 10*idx+delta) & (mapnum < 10*(idx+1)):
                        a.SetDoubleProp('dummyMapNumber', mapnum - delta)
                else:
                    continue
    
        if len(connect_idxs) == 2:
            c1, c2 = connect_idxs
            b = min(bonds)
            rwcombo.AddBond(c1, c2, BONDTYPES[b])
            for c, d in zip(connect_idxs, dummy_idxs):
                rwcombo.RemoveBond(c, d)
            if (dummy_idxs[0] in stereo_atoms) | (dummy_idxs[1] in stereo_atoms):
                idxs = {dummy_idxs[0]: c2, dummy_idxs[1]: c1}
                for bn, aids in enumerate(stereos):
                    stereos[bn] = [idxs[a] if a in idxs.keys() else a for a in aids]

    # assign cis/trans stereo
    i = 0
    for bond in rwcombo.GetBonds():
        if bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE:
            bond.SetStereoAtoms(*stereos[i])
            i += 1

    # remove dummy atoms
    try:
        mol = AllChem.ReplaceSubstructs(rwcombo, Chem.MolFromSmiles('*'), Chem.MolFromSmiles('[H]'), replaceAll= True)[0]
        mol = Chem.RemoveHs(mol, remover, sanitize= False)
        mol = AllChem.ReplaceSubstructs(mol, Chem.MolFromSmiles('[H]'), Chem.MolFromSmiles('*'), replaceAll= True)[0]
        mol = AllChem.DeleteSubstructs(mol, Chem.MolFromSmiles('*'))
        for _ in range(len(fragments)):
            if Chem.SanitizeMol(mol, catchErrors= True) == Chem.rdmolops.SanitizeFlags.SANITIZE_NONE:
                break

        mol.UpdatePropertyCache(strict=True)
        if useChiral:
            chirals = Chem.FindMolChiralCenters(mol)
            if chirals:
                tmpmol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical= True))
                try:
                    tmpmol = Chem.RenumberAtoms(tmpmol, tmpmol.GetSubstructMatch(mol))
                    for (a1, c1), (a2, c2) in zip(chirals, Chem.FindMolChiralCenters(tmpmol)):
                        if (a1 == a2) & (c1 != c2):
                            mol.GetAtomWithIdx(a1).InvertChirality()
                except Exception as e:
                    pass
        # mol = stand.standardize(mol)    # sanitize
        smi = Chem.MolToSmiles(mol, canonical= True)
        tmpmol = deepcopy(mol)
        err = Chem.SanitizeMol(tmpmol, catchErrors= True)
        if err == Chem.rdmolops.SanitizeFlags.SANITIZE_KEKULIZE:
            smi = smi.replace('[n+]', '[n]')
        smi = Chem.CanonSmiles(smi)
    except Exception as e:
        print(str(e), flush= True)
        return None

    if asMol:
        return Chem.MolFromSmiles(smi)
    else:
        return smi
    

def constructMolwithECFP(frag_smiles: list, adj: list, ecfp: list, radius: int):
    smi = constructMol(frag_smiles, adj, asMol= False)
    
    return isomer_search(smi, ecfp, radius)


def constructMolwithTimeout(frag_smiles: list, adj: list, asMol: bool= False, useChiral: bool= True, timeout: int= 300):
    @timeout_decorator.timeout(timeout, use_signals= False)
    def construct(frag_smiles: list, adj: list, asMol: bool= False, useChiral: bool= True):
        return constructMol(frag_smiles, adj, asMol, useChiral)
    try:
        smi = construct(frag_smiles, adj, asMol, useChiral)
    except Exception as e:
        print(e, flush= True)
        smi = None
    return smi


if __name__ == '__main__':
    fragments = ['*C(*)(*)*', '*n1c(=O)n(*)c(=O)n(*)c1=O', '*C(*)=O', '*CC(F)(F)F', '*C(*)=O', '*N*', '*C(*)(*)*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*']
    adj = [ [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    
    smi = constructMol(fragments, adj, validity_correction= False)
    print(smi)