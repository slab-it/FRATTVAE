from copy import deepcopy
import numpy as np
from itertools import chain

from rdkit import Chem, RDLogger
from rdkit.Chem import RWMol, AllChem

from utils.fragmentation import find_BRICSbonds, find_BRICSbonds_and_rings, find_MedChemFrag

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

BONDTYPES = {1: Chem.rdchem.BondType.SINGLE, 
             2: Chem.rdchem.BondType.DOUBLE,
             3: Chem.rdchem.BondType.TRIPLE}


def setAtomMapNumsWithIdxs(mol, indexs= None):
    indexs = indexs if indexs else range(mol.GetNumAtoms())
    for i, atom in zip(indexs, mol.GetAtoms()):
        atom.SetAtomMapNum(i)

def clearAtomMapNums(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

def check_fragSize(mol, minSize: int= 1, maxDegree: int= 32):
    frags = Chem.GetMolFrags(mol, asMols= True, sanitizeFrags= False)
    cut = True
    for f in frags:
        Chem.SanitizeMol(f, catchErrors= True)
        f = Chem.MolFromSmiles(Chem.MolToSmiles(f))
        # check atom num exclude dummy atom
        dummynum = sum([a.GetAtomicNum() == 0 for a in f.GetAtoms()])
        atomnum = f.GetNumAtoms() - dummynum
        if (atomnum < minSize) | (dummynum > maxDegree):
            cut = False
            break

    return cut

def HydrogenMatch(f, f_dash, uniquify: bool= True):
    orders = f_dash.GetSubstructMatches(f, uniquify= uniquify)
    canonOrder = None
    for order in orders:
        f_ordered = Chem.RenumberAtoms(f_dash, order)
        assert f.GetNumAtoms() == f_ordered.GetNumAtoms()
        for a, a_dash in zip(f.GetAtoms(), f_ordered.GetAtoms()):
            if a.GetAtomicNum() != a_dash.GetAtomicNum(): break
            elif a.GetNumExplicitHs() != a_dash.GetNumExplicitHs(): break
        else:
            canonOrder = order
            break

    if (canonOrder is None) & uniquify:
        canonOrder = HydrogenMatch(f, f_dash, uniquify= False)

    if (canonOrder is None) & bool(orders):
        canonOrder = orders[0]
    
    return canonOrder

def MapNumsToAdj(bondMapNums: list, bond_types: list):
    n_frags = len(bondMapNums)
    if n_frags == 1:
        adj = [[0]]
    else:
        n_bonds = len(bond_types)
        adj = [[0] * n_frags for _ in range(len(bondMapNums))]

        for b in range(1, n_bonds+1):
            i1, i2 = [i for i in range(n_frags) if b in bondMapNums[i]]
            adj[i1][i2] = bond_types[b-1]
            adj[i2][i1] = bond_types[b-1]

    return adj


def MolToBRICSfragments(mol, minFragSize: int= 1, maxDegree: int= 32, useChiral: bool= True, useStereo: bool= False):
    rwmol = RWMol(mol)
    numatoms = rwmol.GetNumAtoms()

    # search break bonds
    # matches = find_BRICSbonds(rwmol)
    matches = find_BRICSbonds_and_rings(rwmol)
    # matches = find_MedChemFrag(mol)

    # decompose mol
    bond_types = []
    chiral_centers = []
    stereos = [list(bond.GetStereoAtoms()) for bond in rwmol.GetBonds() if bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE]
    stereo_atoms = list(chain.from_iterable(stereos))
    i = 0
    while i < len(matches):
        match = matches[i]
        a1, a2 = match[0], match[1]
        bond = rwmol.GetBondBetweenAtoms(a1, a2)
        bond_type = int(bond.GetBondTypeAsDouble())

        # stereo chem
        if useStereo & (bond.GetBondType() == BONDTYPES[2]):
            matches.remove(match)
            continue

        # check whether fragment size is bigger than min-size
        tmpmol = deepcopy(rwmol)
        rwmol.RemoveBond(a1, a2)
        for a in match:
            rwmol.AddAtom(Chem.Atom(0))
            rwmol.GetAtomWithIdx(numatoms).SetAtomMapNum(i+1)
            # rwmol.AddBond(a, numatoms, BONDTYPES[1])
            rwmol.AddBond(a, numatoms, BONDTYPES[bond_type])

            numatoms += 1

        if check_fragSize(rwmol.GetMol(), minSize= minFragSize, maxDegree= maxDegree):
            i += 1
            bond_types.append(bond_type)
            chiral_centers += [a for a in match if tmpmol.GetAtomWithIdx(a).HasProp('_CIPCode')]
            # fix cis/trans stereo
            if (a1 in stereo_atoms) | (a2 in stereo_atoms):
                idxs = {a1: numatoms-1, a2: numatoms-2}
                for bn, aids in enumerate(stereos):
                    stereos[bn] = [idxs[a] if a in idxs.keys() else a for a in aids]
        else:
            rwmol = deepcopy(tmpmol)      # role back
            numatoms -= len(match)
            matches.remove(match)

    # assign cis/trans stereo
    i = 0
    for bond in rwmol.GetBonds():
        if bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE:
            try:
                bond.SetStereoAtoms(*stereos[i])
            except:
                pass
            i += 1

    # fragmentation
    fragments_numed = Chem.GetMolFrags(rwmol, asMols= True, sanitizeFrags= False)
    fragments = [Chem.MolFromSmiles(Chem.MolToSmiles(f, canonical= True)) for f in Chem.GetMolFrags(rwmol, asMols= True, sanitizeFrags= False)]

    # only one fragment
    if len(fragments) == 1:
        return [Chem.MolToSmiles(mol)], [0], [[0]]

    # convert to canonical atom orders
    bondMapNums = []
    etc = 1
    for f_numed, f in zip(fragments_numed, fragments):  
        Chem.SanitizeMol(f_numed, catchErrors= True)

        # make connect map
        bondMapNum = sorted([a.GetAtomMapNum() for a in f.GetAtoms() if a.GetAtomMapNum()])
        if len(bondMapNum) == 0:
            bondMapNum = [len(matches)+etc]
            etc += 1
        bondMapNums.append(bondMapNum)

        # reset chirality
        if useChiral:
            canonOrderIdxs = HydrogenMatch(f, f_numed)
            f_renumed = Chem.RenumberAtoms(f_numed, canonOrderIdxs)
            chirals_f = Chem.FindMolChiralCenters(f)
            chirals_f_renumed = Chem.FindMolChiralCenters(f_renumed)
            if len(chirals_f) != len(chirals_f_renumed):
                chirals_nums, _ = zip(*chirals_f) if chirals_f else ([], [])
                for a, ctype in chirals_f_renumed:
                    if a not in chirals_nums:
                        f.GetAtomWithIdx(a).SetProp('_CIPCode', ctype)
                        f.GetAtomWithIdx(a).SetChiralTag(f_renumed.GetAtomWithIdx(a).GetChiralTag())
                Chem.AssignCIPLabels(f)
                chirals_f = Chem.FindMolChiralCenters(f)
            for (a1, c1), (a2, c2) in zip(chirals_f, chirals_f_renumed):
                if (a1 == a2) & (c1 != c2):
                    f.GetAtomWithIdx(a1).InvertChirality()

            # assign cis/trans stereo
            for bond in f_renumed.GetBonds():
                if bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE:
                    a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    try:
                        f.GetBondBetweenAtoms(a1, a2).SetStereo(bond.GetStereo())
                        f.GetBondBetweenAtoms(a1, a2).SetStereoAtoms(*list(bond.GetStereoAtoms()))
                    except:
                        pass

        # numbering dummy atoms
        dummys = [(a.GetAtomMapNum(), a.GetIdx()) for a in f.GetAtoms() if a.GetAtomMapNum()]
        sort_dummys = sorted(dummys, key= lambda dummys: dummys[0])
        for n, (_, i) in enumerate(sort_dummys):
            f.GetAtomWithIdx(i).SetAtomMapNum(n+1)

    frag_smiles = []
    for f in fragments:
        # Chem.SanitizeMol(f)   # miss stereo chemistry
        frag_smiles.append(Chem.MolToSmiles(f).replace('\\\\', '\\'))

    return frag_smiles, bond_types, bondMapNums


def MolFromFragments(frag_smiles: list, adj: list, asMol: bool= False, useChiral: bool= True):
    """ 
    frag_smiles: list of fragments as SMILES,
    adj: torch.Tensor or list shape= (len(frag_smiles), len(frag_smiles)), 0: none, 1: single, 2: double, 3: triple
    """
    if len(frag_smiles) == 1:
        mol = Chem.MolFromSmiles(Chem.CanonSmiles(frag_smiles[0])) if asMol else Chem.CanonSmiles(frag_smiles[0])
        return mol
    
    # remove padding
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
                        assert len(a.GetNeighbors()) == 1
                        neighbor = a.GetNeighbors()[0]
                        dummy_idxs.append(a.GetIdx())
                        connect_idxs.append(neighbor.GetIdx())
                        bonds.append(int(rwcombo.GetBondBetweenAtoms(a.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble()))
                    elif (mapnum > 10*idx+delta) & (mapnum < 10*(idx+1)):
                        a.SetDoubleProp('dummyMapNumber', mapnum - delta)
                else:
                    continue
    
        assert len(connect_idxs) == 2
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
    remover = Chem.RemoveHsParameters()
    remover.removeDegreeZero = False
    mol = AllChem.ReplaceSubstructs(rwcombo, Chem.MolFromSmiles('*'), Chem.MolFromSmiles('[H]'), replaceAll= True)[0]
    mol = Chem.RemoveHs(mol, remover, sanitize= False)
    mol = AllChem.ReplaceSubstructs(mol, Chem.MolFromSmiles('[H]'), Chem.MolFromSmiles('*'), replaceAll= True)[0]
    mol = AllChem.DeleteSubstructs(mol, Chem.MolFromSmiles('*'))
    
    # sanitize fragments
    for _ in range(len(fragments)):
        if Chem.SanitizeMol(mol, catchErrors= True) == Chem.rdmolops.SanitizeFlags.SANITIZE_NONE:
            break
    
    # check chirality
    if useChiral:
        chirals = Chem.FindMolChiralCenters(mol)
        if chirals:
            tmpmol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical= True))
            tmpmol = Chem.RenumberAtoms(tmpmol, tmpmol.GetSubstructMatch(mol))
            for (a1, c1), (a2, c2) in zip(chirals, Chem.FindMolChiralCenters(tmpmol)):
                if (a1 == a2) & (c1 != c2):
                    mol.GetAtomWithIdx(a1).InvertChirality()

    # mol = stand.standardize(mol)
    # Chem.SanitizeMol(mol)
    smi = Chem.MolToSmiles(mol, canonical= True)

    if asMol:
        return Chem.MolFromSmiles(smi)
    else:
        return smi
    

if __name__ == '__main__':
    smi = 'COc1ccccc1/C=C/C=C(\C#N)C(=O)Nc1ccc(C(=O)N(C)C)cc1'
    frag_smiles, bond_types, bondMapNums = MolToBRICSfragments(Chem.MolFromSmiles(smi), useStereo= True)
    # n_frags = len(frag_smiles)
    # tmp_ecfps = [[0, 0] for _ in range(len(frag_smiles))]
    # tree = make_DFStree(list(range(n_frags)), tmp_ecfps, bond_types, bondMapNums)
    # fids = tree.dgl_graph.ndata['fid'].squeeze(-1)
    # adj = tree.adjacency_matrix().to_dense()
    # frag_smiles = [frag_smiles[fid] for fid in fids]
    adj = MapNumsToAdj(bondMapNums, bond_types)
    smi_rev = MolFromFragments(frag_smiles, adj)
    print(smi)
    print(smi_rev)