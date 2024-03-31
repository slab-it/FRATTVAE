import os
import argparse

from rdkit import Chem
import pandas as pd
import yaml
import datetime

class DictProcessor(argparse.Action):
    def __call__(self, parser, namespace, values, option_strings=None):
        param_dict = getattr(namespace,self.dest,[])
        if param_dict is None:
            param_dict = {}

        k, v = values.split(":")
        param_dict[k] = int(v)
        setattr(namespace, self.dest, param_dict)

# set parser
parser = argparse.ArgumentParser(description= 'please enter paths')
parser.add_argument('data_path', type= str, help= 'csv file')
parser.add_argument('--seed', type= int, default= 0, help= 'random seed')
parser.add_argument('--exist_dir', action= 'store_true')

# decompose
parser.add_argument('--maxLength', type= int, default= 32, help= 'max number of fragments per a mol')
parser.add_argument('--maxDegree', type= int, default= 16, help= 'max number of degree per a fragment')
parser.add_argument('--minSize', type= int, default= 1, help= 'min number of atoms per a fragment')
parser.add_argument('--ecfpBits', type= int, default= 2048, help= 'number of ecfp bits')
parser.add_argument('--ecfpRadius', type= int, default= 2, help= 'number of ecfp bits')
parser.add_argument('--useChiral', type= int, default= 1, help= 'use chirality')
parser.add_argument('--breakDouble', action= 'store_false')
parser.add_argument('--ignoreDummy', action= 'store_true')

# model
parser.add_argument('--d_latent', type= int, default= 256, help= 'd_latent')
parser.add_argument('--d_model', type= int, default= 512, help= 'd_model')
parser.add_argument('--d_ff', type= int, default= 2048, help= 'd_ff')
parser.add_argument('--nlayer', type= int, default= 6, help= 'num_layers')
parser.add_argument('--nhead', type= int, default= 8, help= 'num_heads')
parser.add_argument('--dropout', type= float, default= 0.1, help= 'dropout')
parser.add_argument('--activation', type= str, default= 'gelu', choices= ['relu', 'gelu'], help= "activation func. please choice from ['relu', 'gelu']")

# condtional
parser.add_argument('--condition', default= {}, action= DictProcessor, help= 'condition column name: num_categories. if condition is continuous value, num_categories = 1. ex. QED:1')

# property prediction
parser.add_argument('--property', default= {}, action= DictProcessor, help= 'property column name: num_categories. if the task is regression or binary classification, num_categories = 1. ex. QED:1')
parser.add_argument('--ploss', type= str, default= 'bce', choices= ['bce', 'mse', 'mae', 'crs'], help= "property loss. please choice from ['bce', 'mse', 'mae', 'crs']")

# trainning
parser.add_argument('--epoch', type= int, default= 1000, help= 'num_epoch')
parser.add_argument('--batch_size', type= int, default= 512, help= 'batch_size. if multi gpu are used, batch_size par one gpu is batch_size//n_gpu')
parser.add_argument('--lr', type= float, default= 0.0001, help= 'learning rate')
parser.add_argument('--kl_w', type= float, default= 0.0005, help= 'weight for kl-divergence')
parser.add_argument('--anneal_epoch', action= DictProcessor, help= 'update kl_weight to kl_weight * value at the epoch. please enter epoch:value. ex. 400:2')
parser.add_argument('--l_w', type= float, default= 2.0, help= 'weight for label prediction loss')
parser.add_argument('--p_w', type= float, default= 1.0, help= 'weight for property prediction loss')
args = parser.parse_args()

if bool(args.condition) & bool(args.property):
    raise ValueError('please choose conditions or properties')
elif bool(args.condition):
    task = 'condition'
    props = args.condition
elif bool(args.property):
    task = 'property'
    props = args.property
else:
    task = 'struct'
    props = {}
pnames = list(props.keys())
spl = '_' if bool(pnames) else ''

df = pd.read_csv(args.data_path)
for pname in pnames:
    if pname not in df.columns:
        raise ValueError(f'{pname} is not in columns of DataFrame.')
print(f'loaded: {args.data_path}, smiles: {len(df)}', flush= True)

# remove chirality
if args.useChiral == 0:
    df['SMILES'] = [Chem.CanonSmiles(s, useChiral= args.useChiral) for s in df.SMILES]

dname = args.data_path.split('/')[-1].split('.')[0]
today = datetime.datetime.today()
result_path = os.path.join('../results', f'{dname}_{task}{spl}{"_".join(pnames)}_{today.month:0>2}{today.day:0>2}')
result_path = os.path.abspath(result_path)

os.makedirs(os.path.join(result_path, 'input_data'), exist_ok= args.exist_dir)
os.makedirs(os.path.join(result_path, 'models'), exist_ok= args.exist_dir)
os.makedirs(os.path.join(result_path, 'train'), exist_ok= args.exist_dir)
os.makedirs(os.path.join(result_path, 'valid'), exist_ok= args.exist_dir)
os.makedirs(os.path.join(result_path, 'test'), exist_ok= args.exist_dir)
os.makedirs(os.path.join(result_path, 'generate'), exist_ok= args.exist_dir)
os.makedirs(os.path.join(result_path, 'visualize'), exist_ok= args.exist_dir)
print('makedirs: ' + result_path, flush= True)

# make yaml
dupl_bits = 0
anneal_epochs = {int(key): value for key, value in args.anneal_epoch.items()} if args.anneal_epoch else {}
with open(os.path.join(result_path, 'input_data', 'params.yml'), 'w') as yf:
    yf.write(f'# Date of creation: {datetime.datetime.now()}\n')
    yaml.dump({
        'result_path': result_path, 
        'data_path': args.data_path,
        'frag_path': None,
        'seed': args.seed,

        'decomp': {
            'min_size': args.minSize,
            'max_nfrags': args.maxLength,
            'n_bits': args.ecfpBits,
            'dupl_bits': dupl_bits,
            'radius': args.ecfpRadius,
            'max_depth': args.maxLength,
            'max_degree': args.maxDegree,
            'useChiral': bool(args.useChiral),
            'ignore_double': args.breakDouble,
            'ignore_dummy': args.ignoreDummy,
        },

        'model': {
            'd_model': args.d_model,
            'd_ff': args.d_ff,
            'nlayer': args.nlayer,
            'nhead': args.nhead,
            'latent': args.d_latent,
            'feat': args.ecfpBits + dupl_bits,
            'property': props,
            'ploss': args.ploss,
            'activation': args.activation,
            'dropout': args.dropout,
        },

        'train': {
            'epoch': args.epoch,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'kl_w': args.kl_w,
            'kl_anneal': anneal_epochs,   # key= epoch, kl_w * value
            'l_w': args.l_w,
            'p_w': args.p_w,
        }
        }, yf, default_flow_style= False, sort_keys= False)
    
print(f"Config is saved to {os.path.join(result_path, 'input_data', 'params.yml')}", flush= True)