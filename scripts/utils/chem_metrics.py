from typing import List

import random
import numpy as np
from rdkit import Chem
from moses import metrics
import networkx as nx

from guacamol.distribution_matching_generator import DistributionMatchingGenerator
from guacamol.distribution_learning_benchmark import KLDivBenchmark
from guacamol.frechet_benchmark import FrechetBenchmark


def normalize(props: np.ndarray, pname: str):
    if pname not in NORM_PARAMS.keys():
        return props.tolist()
    else:
        low, high = NORM_PARAMS[pname]
        return (props - low) / (high - low)
    

def get_all_metrics(mol) -> dict:
    if type(mol) == str:
        mol = Chem.MolFromSmiles(mol)
    if mol is None:
        return [float('nan')] * len(METRICS_DICT)
    return [func(mol) for func in METRICS_DICT.values()]


def get_metrics(mol, key: str) -> dict:
    if type(mol) == str:
        mol = Chem.MolFromSmiles(mol)
    if mol is None:
        return float('nan')
    return METRICS_DICT[key](mol)
    

class MockGenerator(DistributionMatchingGenerator):
    """
    Mock generator that returns pre-defined molecules,
    possibly split in several calls
    """
    def __init__(self, molecules: List[str]) -> None:
        self.molecules = molecules
        self.cursor = 0

    def generate(self, number_samples: int) -> List[str]:
        end = self.cursor + number_samples

        sampled_molecules = self.molecules[self.cursor:end]
        self.cursor = end
        return sampled_molecules


def physchem_divergence(smiles: list, reference: list, seed: int= 0):
    """
    smiles: list of generated smiles
    reference: list of training/test smiles

    return average physchem kl-divergence
    """
    generator = MockGenerator(smiles)
    if len(reference) < len(smiles):
        random.seed(seed)
        reference += random.choices(reference, k= len(smiles)-len(reference))
    benchmark = KLDivBenchmark(number_samples= len(smiles), training_set= reference)

    return benchmark.assess_model(generator).score


def guacamol_fcd(smiles: list, reference: list, seed: int= 0):
    """
    smiles: list of generated smiles
    reference: list of training/test smiles

    rerurn standardize fcd [exp(-0.2 * fcd)]
    """
    generator = MockGenerator(smiles)
    if len(reference) < len(smiles):
        random.seed(seed)
        reference += random.choices(reference, k= len(smiles)-len(reference))
    benchmark = FrechetBenchmark(sample_size= len(smiles), training_set= reference)
    try:
        score = benchmark.assess_model(generator).score
        return score
    except Exception as e:
        print(f'[WARNING] can\'t calculate fcd score because of {e}...', flush= True)
        return float('nan')


def penalized_logp(mol):
    """
    Reward that consists of log p penalized by SA and # long cycles,
    as described in (Kusner et al. 2017). Scores are normalized based on the
    statistics of 250k_rndm_zinc_drugs_clean.smi dataset
    :param mol: rdkit mol object
    :return: float

    Reference: 'https://github.com/bowenliu16/rl_graph_generation/blob/master/gym-molecule/gym_molecule/envs/molecule.py'
    """
    # normalization constants, statistics from 250k_rndm_zinc_drugs_clean.smi
    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    log_p = metrics.logP(mol)
    SA = -metrics.SA(mol)

    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(
        Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length

    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std

    return normalized_log_p + normalized_SA + normalized_cycle


NORM_PARAMS = {'MW': [0, 1000], 
               'logP': [-10, 10], 
               'QED': [0, 1], 
               'SA': [10, 0], 
               'NP': [-5, 5],
               'TPSA': [0, 300],
            #    'BertzCT': [0, 2000],
            #    'PlogP': [-20, 0]
               }

METRICS_DICT = {'MW': metrics.weight,
                'logP': metrics.logP,
                'QED': metrics.QED,
                'SA': metrics.SA,
                'NP': metrics.NP,
                'TPSA': Chem.Descriptors.TPSA,
                # 'BertzCT': Chem.Descriptors.BertzCT,
                # 'PlogP': penalized_logp
                }