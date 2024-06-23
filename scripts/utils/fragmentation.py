import os
from itertools import chain
from rdkit import Chem
from rdkit.Chem import RWMol, BRICS

from utils.medchemfrag import decomposition


def FindBRICS(mol, bonds: list= None, AtomDone: set= None):
    bonds = bonds if bonds is not None else []
    AtomDone = AtomDone if AtomDone is not None else set([])

    for idxs, _ in BRICS.FindBRICSBonds(mol):
        idxs = sorted(idxs)
        if (idxs in bonds) or (len(set(idxs) & AtomDone) > 1):  # both atoms have already been searched.
            continue
        else:
            bonds.append(sorted(idxs))
            AtomDone = AtomDone | set(idxs)
    return bonds, AtomDone


def FindRings(mol, bonds: list= None, AtomDone: set= None):
    bonds = bonds if bonds is not None else []
    AtomDone = AtomDone if AtomDone is not None else set([])
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()

        if (begin.GetIdx() in AtomDone) & (end.GetIdx() in AtomDone):
            continue

        # bond between rings, ring and C-ANY
        if (begin.IsInRing() | end.IsInRing()) & (not bond.IsInRing()) & (bond.GetBondTypeAsDouble() < 2):
            if begin.IsInRing() & end.IsInRing():
                neighbor = 1
            elif begin.IsInRing():
                neighbor = len(end.GetNeighbors()) - 1
            elif end.IsInRing():
                neighbor = len(begin.GetNeighbors()) - 1
            else:
                neighbor = 0
            
            if neighbor > 0:
                idxs = sorted([begin.GetIdx(), end.GetIdx()])
                if idxs not in bonds:
                    bonds.append(idxs)
                    AtomDone = AtomDone | set(idxs)
    return bonds, AtomDone


# def FindStereo(mol, bonds: list= None):
#     """
#     find cis/trans stereo type
#     """
#     bonds = bonds if bonds is not None else []
#     for bond in mol.GetBonds():
#         if bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE:
#             stereo_atoms = list(bond.GetStereoAtoms())
#             atom_idxs = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
#             for aid in atom_idxs:
#                 neighbors = mol.GetAtomWithIdx(aid).GetNeighbors()
#                 if len(neighbors) > 2:
#                     idx = [n.GetIdx() for n in neighbors if (n.GetIdx() not in stereo_atoms) & (n.GetIdx() not in atom_idxs)][0]
#                     idxs = sorted([aid, idx])
#                     if idxs not in bonds:
#                         bonds.append(idxs)
#     return bonds

def find_BRICSbonds(mol) -> list:
    return sorted([sorted(idxs) for idxs, _ in BRICS.FindBRICSBonds(mol)], key= lambda idxs: idxs[1])

def find_rings(mol) -> list:
    return sorted(FindRings(mol)[0], key= lambda idxs: idxs[1])

def find_MedChemFrag(mol) -> list:
    return sorted([sorted(idxs) for idxs in decomposition(mol)], key= lambda idxs: idxs[1])

def find_BRICSbonds_and_rings(mol) -> list:
    """
    Find bonds which are BRICS bonds or single bonds between rings.
    """
    # bonds, AtomDone = FindBRICS(mol)
    bonds = find_BRICSbonds(mol)
    return sorted(FindRings(mol, bonds= bonds)[0], key= lambda idxs: idxs[1])
