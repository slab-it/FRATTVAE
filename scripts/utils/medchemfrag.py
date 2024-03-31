from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

"""
Reference: http://molmodel.com/hg/medchem_fragment_splitter (2023/02/13)
"""

rules = [
    "[R]-!@[$([CX4;H2,H1,H0])]",  # 1
    "[a]-!@[$([NX3;H1,H0]),$([OX2;H0]),$([SX2;H0])]-!@[$([C;H2,H1,H0]);!$([CX3]=[OX1])]",  # 2
    "[a]-!@[$([NX3;H1,H0]),$([OX2;H0]),$([SX2;H0])]-!@[a]",  # 3
    "[a]-!@[$([CX3]=[OX1,NX2,SX1,CX3])]-!@[$([CX4;H2,H1,H0])]",  # 4
    "[c]-!@[$([CX3]=[OX1,NX2,SX1,$([CX3;H2])])]-!@[c]",  # 5.1
    "[n]-!@[$([CX3]=[OX1,NX2,SX1,$([CX3;H2])])]-!@[c]",  # 5.2
    "[$([CX4;H2,H1,H0])]-!@[CX3](=[OX1])[OX2;H0]",  # 6
    "[$([CX4;H2,H1,H0])]-!@[OX2;H0][CX3](=[OX1])",  # 7
    "[a]-!@[CX3](=[OX1])O-!@[$([CX4;H2,H1,H0])]",  # 8
    "[a]-!@[CX3](=[OX1])O-!@[a]",  # 9
    "[a]-!@[NX2;H0]=[NX2;H0]-!@[$([CX4;H2,H1,H0])]",  # 10
    "[a]-!@[NX2;H0]=[NX2;H0]-!@[a]",  # 11
    "[a]-!@[NX3;H1]-!@[$([CX3;H0](=[OX1]))]-!@[$([CX4;H2,H1,H0])]",  # 12
    "[a]-!@[$([CX3;H0](=[OX1]))]-!@[NX3;H1]-!@[$([CX4;H2,H1,H0])]",  # 13
    "[a]-!@[NX3;H1]-!@[$([CX3;H0](=[OX1]))]-!@[a]",  # 14
    "[a]-!@[$([CX3;H0](=[OX1]))]-!@[NX3;H1]-!@[a]",  # 15
    "[a]-!@[SX4](=[OX1])(=[OX1])[NX3;H1]-!@[$([CX4;H2,H1,H0])]",  # 16
    "[a]-!@[NX3;H1][SX4](=[OX1])(=[OX1])-!@[$([CX4;H2,H1,H0])]",  # 17
    "[a]-!@[SX4](=[OX1])(=[OX1])[NX3;H1]-!@[a]",  # 18
    "[a]-!@[NX3;H1][SX4](=[OX1])(=[OX1])-!@[NX3;H1]-!@[a]",  # 19
    "[$([CX4;H2,H1,H0])]-!@[NX3][CX3](=[OX1])",  # 20
    "[$([CX4;H2,H1,H0])]-!@[CX3](=[OX1])[NX3]",  # 21
    "[$([CX4;H2,H1,H0])]-!@[$([NX3;H1,H0])]",  # 22
    "[$([CX4;H2,H1,H0])]-!@[$([OX2;H0])]",  # 23
    "[$([CX4;H2,H1,H0])]-!@[$([SX2;H0])]",  # 24
    "[$([CX4;H2,H1,H0])]-!@[SX4](=[OX1])(=[OX1])[NX3;H1]",  # 25
    "[$([CX4;H2,H1,H0])]-!@[NX3;H1][SX4](=[OX1])(=[OX1])"  # 26
]

index_all = [
    [(0, 1)],  # 1
    [(1, 2)],  # 2
    [(0, 1), (1, 2)],  # 3
    [(1, 2)],  # 4
    [(0, 1), (1, 2)],  # 5.1
    [(1, 2)],  # 5.2
    [(0, 1)],  # 6
    [(0, 1)],  # 7
    [(3, 4)],  # 8
    [(0, 1), (3, 4)],  # 9
    [(2, 3)],  # 10
    [(0, 1), (2, 3)],  # 11
    [(2, 3)],  # 12
    [(2, 3)],  # 13
    [(0, 1), (2, 3)],  # 14
    [(0, 1), (2, 3)],  # 15
    [(4, 5)],  # 16
    [(4, 5)],  # 17
    [(0, 1), (4, 5)],  # 18
    [(0, 1), (4, 5)],  # 19
    [(0, 1)],  # 20
    [(0, 1)],  # 21
    [(0, 1)],  # 22
    [(0, 1)],  # 23
    [(0, 1)],  # 24
    [(0, 1)],  # 25
    [(0, 1)],  # 26
]

def decomposition(molecule, smarts: list= None):
    only_smarts = (smarts is None)

    atomPairs = []
    if not only_smarts:
        for index in range(len(smarts)):
            smartFragment = Chem.MolFromSmiles(smarts[index])
            atomPairsLoc = molecule.GetSubstructMatches(smartFragment)
            for atomPair in atomPairsLoc:
                atomPairs.append(atomPair)
    else:
        for index in range(len(rules)):
            atomPairsLoc = molecule.GetSubstructMatches(Chem.MolFromSmarts(rules[index]))
            for nb in range(len(index_all[index])):
                index1 = index_all[index][nb][0]
                index2 = index_all[index][nb][1]
                for atomPair in atomPairsLoc:
                    atomIndex1 = atomPair[index1]
                    atomIndex2 = atomPair[index2]
                    atomPairs.append((atomIndex1, atomIndex2))

    if (len(atomPairs) == 0):
        return []
    
    atomPairs = list(set(atomPairs))
    bonds = list()
    if not only_smarts:
        flag = False
        for atomIndexes1 in atomPairs:
            for atomIndexes2 in atomPairs:
                if atomIndexes1 == atomIndexes2:
                    continue
                else:
                    for a1 in atomIndexes1:
                        for a2 in atomIndexes2:
                            if a1 == a2:
                                flag = True
                                if len(atomIndexes1) > len(atomIndexes2):
                                    if atomIndexes2 in atomPairs:
                                        atomPairs.remove(atomIndexes2)
                                else:
                                    if atomIndexes1 in atomPairs:
                                        atomPairs.remove(atomIndexes1)
                                    break
                        if flag:
                            break
                flag = False

    for atomIndexes in atomPairs:
        for a1 in atomIndexes:
            for a2 in atomIndexes:
                if a1 == a2:
                    continue
                else:
                    bond = molecule.GetBondBetweenAtoms(a1, a2)
                    if (bond != None):
                        idxs = sorted([a1, a2])
                        if idxs not in bonds:
                            bonds.append(idxs)
    # bonds = list(set(bonds))
    return bonds


def add_nitrogen_charges(m):
    m.UpdatePropertyCache(strict=False)
    ps = Chem.DetectChemistryProblems(m)
    if not ps:
        Chem.SanitizeMol(m)
        return m
    for p in ps:
        if p.GetType()=='AtomValenceException':
            at = m.GetAtomWithIdx(p.GetAtomIdx())
            if at.GetAtomicNum()==7 and at.GetFormalCharge()==0 and at.GetExplicitValence()==4:
                at.SetFormalCharge(1)
    Chem.SanitizeMol(m)
    return m