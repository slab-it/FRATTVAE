import os
import argparse
import gc
import time
import datetime
import json
import yaml
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

import pandas as pd
import numpy as np
from rdkit import Chem

import torch
import pickle
from functools import partial

from mso.optimizer import BasePSOptimizer, ParallelSwarmOptimizer
from mso.objectives.scoring import ScoringFunction
from mso.objectives.mol_functions import qed_score, penalized_logp_score, tan_sim, sa_score
from guacamol.assess_goal_directed_generation import assess_goal_directed_generation

from utils.apps import second2date, torch_fix_seed
from inference import InferenceModel, GuacaGoalInference

# set parser
parser = argparse.ArgumentParser(description= 'please enter paths')
parser.add_argument('yml', type= str, help= 'yml file')
parser.add_argument('--load_epoch', type= int, default= None, help= 'load model at load epoch')
parser.add_argument('--x_max', type= float, default= None, help= 'max bound of the optimization. default is caluculted from data')
parser.add_argument('--x_min', type= float, default= None, help= 'max bound of the optimization. default is caluculted from data')
parser.add_argument('--num_restart', type= int, default= 1, help= 'the number of optimazation restarts')
parser.add_argument('--num_iters', type= int, default= 10, help= 'the number of optimazation iterations.')
parser.add_argument('--num_swarms', type= int, default= 10, help= 'the number of start population.')
parser.add_argument('--num_part', type= int, default= 100, help= 'the number of partial population')
parser.add_argument('--min_desire', type= float, default= 0.0, help= 'min desirability')
parser.add_argument('--max_desire', type= float, default= 1.0, help= 'max desirability')
parser.add_argument('--gpu', type= int, default= 0, help= 'gpu device ids')
parser.add_argument('--n_jobs', type= int, default= 1, help= 'the number of cpu for parallel, default 24')
parser.add_argument('--seed', type= int, default= 0, help= 'random seed')

# guacamol
parser.add_argument('--guaca_version', type= str, default= 'v2', choices= ['v1', 'v2'], help= 'guacamol version')

# property optimization
parser.add_argument('--scoring_func', type= str, default= None, help= 'scoring_funcion, [qed, plogp]')
parser.add_argument('--num_optim', type= int, default= 10000, help= 'the number of partial population')
parser.add_argument('--init_smiles', type= str, default= None, help= 'init_smiles, ex. c1ccccc1')
parser.add_argument('--smiles_file', type= str, default= None, help= 'smiles txt file')
args = parser.parse_args()

def reverse_sa_score(mol):
    return (10 - sa_score(mol)) / 10

def scoring_func_thr(mol, ref_smiles, scoring_func, thr: float= 0.5):
    similar = tan_sim(mol, ref_smiles)
    if similar >= thr:
        return thr + (1-thr) * scoring_func(mol)
    else:
        return similar

SCORERING = {'qed': qed_score, 'plogp': penalized_logp_score, 'sa': reverse_sa_score}
assert (args.scoring_func is None) | (args.scoring_func in SCORERING.keys())

start = time.time()

# environment
if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.set_device(args.gpu)
else:
    device = 'cpu'
torch_fix_seed(args.seed)
print(f'GPU [{args.gpu}] is available: {torch.cuda.is_available()}\n', flush= True)

yml_file = args.yml
with open(yml_file) as yml:
    params = yaml.safe_load(yml)
result_path= params['result_path']
print(f'load: {yml_file}', flush= True)

infer_model = InferenceModel(yml_file= args.yml, load_epoch= args.load_epoch, device= device, n_jobs= args.n_jobs)

if (args.x_max is None) | (args.x_min is None):
    load_epoch = args.load_epoch if args.load_epoch else '_best'
    with open(os.path.join(result_path, 'train', f'z_list{load_epoch}.pkl'), 'rb') as f:
        z = np.array(pickle.load(f))
    z_max, z_min = z.max(axis = 0), z.min(axis= 0)
    x_max = z_max.mean() + z_max.std()
    x_min = z_min.mean() - z_min.std()
    del z
    gc.collect()
else:
    x_max, x_min = args.x_max, args.x_min

config = {'restart': args.num_restart, 'iter': args.num_iters, 'n_swarms': args.num_swarms, 
          'n_part': args.num_part, 'x_max': x_max, 'x_min': x_min}
print(', '.join([f'{key}: {value}' for key, value in config.items()]), flush= True)
print(f'desirability: [{args.min_desire}, {args.max_desire}]\n', flush= True)

# guacamol
if args.scoring_func is None:
    print(f'---{datetime.datetime.now()}: GuacaMol Goal Directed-{args.guaca_version} start.---', flush= True)
    start = time.time()
    assess_goal_directed_generation(GuacaGoalInference(infer_model, config), 
                                    json_output_file= os.path.join(result_path, 'generate', f'output_goal_directed_{args.guaca_version}_{args.load_epoch}.json'),
                                    benchmark_version= args.guaca_version)
    with open(os.path.join(result_path, 'generate', f'output_goal_directed_{args.guaca_version}_{args.load_epoch}.json')) as f:
        results = json.load(f)['results']
    scores = []
    for res in results:
        scores.append(res["score"])
        print(f'- {res["benchmark_name"]}: {res["score"]:.4f} (elapsed time: {res["execution_time"]:.4f})', flush= True)
    print(f'Sum: {sum(scores):.4f}, Average: {np.mean(scores):.4f}\n', flush= True)
    
# property optimization
elif args.smiles_file is None:
    print(f'---{datetime.datetime.now()}: {args.scoring_func.upper()} Optimization start.---', flush= True)
    desire = [{"x": args.min_desire, "y": args.min_desire}, {"x": args.max_desire, "y": args.max_desire}]
    scoring_func = SCORERING[args.scoring_func]
    scoring_functions = [ScoringFunction(func= scoring_func, name= args.scoring_func, desirability= desire, is_mol_func= True)]
    df_opt = pd.DataFrame(columns=["smiles", "fitness"])
    for i in range(config['restart']):
        init_smiles = infer_model.random_generate(config['n_swarms']) if args.init_smiles is None else args.init_smiles
        print(f'Init Smiles: {init_smiles}', flush= True)
        opt = ParallelSwarmOptimizer.from_query(init_smiles= init_smiles,
                                                num_part= config['n_part'],
                                                num_swarms= len(init_smiles),
                                                inference_model= infer_model,
                                                scoring_functions= scoring_functions,
                                                x_max= x_max,
                                                x_min= x_min
                                                )
        opt.run(config['iter'], num_track= args.num_optim)
        opt_smiles = opt.best_solutions['smiles'][0]
        df_opt = pd.concat([df_opt, opt.best_solutions])
        print(f'[{i+1}/{config["restart"]}] Optimized Smiles: {opt_smiles} ({scoring_func(Chem.MolFromSmiles(opt_smiles)):.4f}), elapsed time: {second2date(time.time()-start)} s', flush= True)
    df_opt = df_opt.sort_values(by= 'fitness', ascending= False).reset_index(drop= True)
    df_opt.to_csv(os.path.join(result_path, 'generate', f'optimize_{args.scoring_func}_{args.load_epoch}.csv'), index= False)
    for s in df_opt['smiles'].iloc[:3]:
        print(f'- {s} ({scoring_func(Chem.MolFromSmiles(s)):.4f})',  flush= True)

# constrained property optimization
else:
    print(f'---{datetime.datetime.now()}: Constrained {args.scoring_func.upper()} Optimization start.---', flush= True)
    init_smiles = [line.rstrip('\n') for line in open(args.smiles_file, 'r').readlines()]
    desire = [{"x": args.min_desire, "y": args.min_desire}, {"x": args.max_desire, "y": args.max_desire}]
    scoring_func = SCORERING[args.scoring_func]

    thrs = [0.25, 0.50]
    for thr in thrs:
        results = {key: [] for key in ['SMILES', 'OPTIMIZED', 'tanimoto', 'score', 'improved']}

        for i, smiles in enumerate(init_smiles):
            partial_sim = partial(tan_sim, ref_smiles= smiles)
            partial_func = partial(scoring_func_thr, ref_smiles= smiles, scoring_func= scoring_func, thr= thr)
            scoring_functions = [ScoringFunction(func= partial_func, name= args.scoring_func, desirability= desire, is_mol_func= True)]

            opt = ParallelSwarmOptimizer.from_query(init_smiles= init_smiles,
                                                    num_part= config['n_part'],
                                                    num_swarms= config['n_swarms'],
                                                    inference_model= infer_model,
                                                    scoring_functions= scoring_functions,
                                                    x_max= x_max,
                                                    x_min= x_min
                                                    )
            opt.run(config['iter'], num_track= config['n_part']*config['n_swarms'])
            opt_smiles = opt.best_solutions['smiles'].tolist()
            scores, similarities = [], []
            for s in opt_smiles:
                m = Chem.MolFromSmiles(s)
                scores.append(scoring_func(m))
                similarities.append(partial_sim(m))

            tmp = np.where(np.array(similarities)>=thr, np.array(scores), 0)
            best_idx = np.argmax(tmp)
            results['SMILES'].append(smiles)
            results['OPTIMIZED'].append(opt_smiles[best_idx])
            results['tanimoto'].append(similarities[best_idx])
            results['score'].append(tmp[best_idx])
            results['improved'].append(tmp[best_idx] - scoring_func(Chem.MolFromSmiles(smiles)))

            if (i==0) | (((i+1) % 10) == 0):
                df_res = pd.DataFrame(results)
                df_res.to_csv(os.path.join(result_path, 'generate', f'const_optimize_{args.scoring_func}_{args.load_epoch}_{int(thr*10)}.csv'), index= False)
                print(f'[{i+1}/{len(init_smiles)}] Average improved: {df_res.improved.loc[df_res.improved>0].mean():.4f} (std; {df_res.improved.loc[df_res.improved>0].std():.4f}), Success rate: {sum(df_res.improved>0)/len(df_res):.4f}, elapsed time: {int(time.time()-start)} sec', flush= True)

        df_res = pd.DataFrame(results)
        df_res.to_csv(os.path.join(result_path, 'generate', f'const_optimize_{args.scoring_func}_{args.load_epoch}_{int(thr*10)}.csv'), index= False)
        print(f'[thr-{thr}] Average improved: {df_res.improved.loc[df_res.improved>0].mean():.4f} (std; {df_res.improved.loc[df_res.improved>0].std():.4f}), Success rate: {sum(df_res.improved>0)/len(df_res):.4f}\n', flush= True)

print(f'---{datetime.datetime.now()}: all process done. (elapsed time: {second2date(time.time()-start)})---\n', flush= True)