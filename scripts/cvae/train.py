import argparse
import os
import datetime
import gc
import pickle
import sys
import time
import warnings
import yaml
warnings.simplefilter('ignore')

# set parser
parser = argparse.ArgumentParser(description= 'please enter paths')
parser.add_argument('yml', type= str, help= 'yml file')
parser.add_argument('--gpus', type= int, default= [1], nargs='*', help= 'a list of gpu device ids. if len(ids) > 1, use DDP')
parser.add_argument('--n_jobs', type= int, default= 1, help= 'the number of cpu for parallel, default 24')
parser.add_argument('--load_epoch', type= int, default= 0, help= 'load model at load epoch, default epoch= 0')
parser.add_argument('--save_interval', type= int, default= 200, help= 'save model every N epochs, default N= 20')
parser.add_argument('--select_metric', type= int, default= 3, help= 'metric to select best model. 0: total, 1: kl, 2: label loss, 3: label acc, default 3')
parser.add_argument('--valid', action= 'store_true')
parser.add_argument('--master_port', type= str, default= '12356', help= 'if you use DDP, select master port. default= "12356"')
parser.add_argument('--seed', type= str, default= 0, help= 'random seed.')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.gpus))

import numpy as np
import pandas as pd
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from models.fttvae import FTTVAE
from models.wrapper import CVAEwrapper
from utils.apps import second2date, torch_fix_seed
from utils.data import collate_pad_fn
from utils.mask import create_mask
from utils.metrics import batched_kl_divergence
from utils.preprocess import SmilesToMorganFingetPrints


def list2pdData(loss_list: list, metrics: list) -> pd.DataFrame:
    loss_dict = {}
    for key, values in zip(metrics, loss_list):
        loss_dict[key] = values
    return pd.DataFrame(loss_dict)

def train(rank, 
          yml_file: str,
          load_epoch: int= 0,
          save_epoch: int= 20,
          validation: bool= True,
          n_jobs: int= 1,
          seed: int= 0,
          ) -> torch.Tensor:
    torch_fix_seed(seed)

    # set device
    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.set_device(rank)
        n_gpu = torch.cuda.device_count()
    else:
        device = 'cpu'
        n_gpu = 1
    multigpu = n_gpu > 1
    if multigpu:
        dist.init_process_group(backend= "nccl", init_method='env://', rank= rank, world_size= n_gpu)

    ## load hyperparameters
    if rank == 0:
        print(f'---{datetime.datetime.now()}: Loading data. ---', flush= True)
        print(f'load: {yml_file}\n', flush= True)

    with open(yml_file) as yml:
        params = yaml.safe_load(yml)

    # path
    result_path= params['result_path']
    data_path = params['data_path']
    frag_path = params['frag_path']

    # hyperparameters for decomposition and tree-fragments
    decomp_params = params['decomp']
    n_bits = decomp_params['n_bits']
    dupl_bits = decomp_params['dupl_bits']
    radius = decomp_params['radius']
    max_depth = decomp_params['max_depth']
    max_degree = decomp_params['max_degree']
    useChiral = decomp_params['useChiral']
    ignore_dummy = decomp_params['ignore_dummy']

    # hyperparameters for model
    model_params = params['model']
    d_model = model_params['d_model']
    d_ff = model_params['d_ff']
    num_layers = model_params['nlayer']
    num_heads = model_params['nhead']
    activation = model_params['activation']
    latent_dim = model_params['latent']
    feat_dim = model_params['feat']
    props = model_params['property']
    pnames = list(props.keys())
    dropout = model_params['dropout']

    # hyperparameters for training
    train_params = params['train']
    epochs = train_params['epoch']
    batch_size = train_params['batch_size'] // n_gpu if train_params['batch_size'] > n_gpu else 1
    lr = train_params['lr']
    kl_w = train_params['kl_w']
    l_w = train_params['l_w']
    

    ## load data
    s = time.time()
    df = pd.read_csv(data_path)

    df_frag = pd.read_csv(frag_path)
    uni_fragments = df_frag.SMILES.tolist()
    freq_label = df_frag['frequency'].tolist()
    with open(os.path.join(result_path, 'input_data', 'dataset.pkl'), 'rb') as f:
        dataset = pickle.load(f)
    try:
        with open(os.path.join(result_path, 'input_data', 'csr_ecfps.pkl'), 'rb') as f:
            frag_ecfps = pickle.load(f).toarray()
            frag_ecfps = torch.from_numpy(frag_ecfps).float()
        assert frag_ecfps.shape[0] == len(uni_fragments)
        assert frag_ecfps.shape[1] == (n_bits + dupl_bits)
    except Exception as e:
        if rank==0: print(e, flush= True)
        frag_ecfps = torch.tensor(SmilesToMorganFingetPrints(uni_fragments[1:], n_bits= n_bits, dupl_bits= dupl_bits, radius= radius, 
                                                            ignore_dummy= ignore_dummy, useChiral= useChiral, n_jobs= n_jobs)).float()
        frag_ecfps = torch.vstack([frag_ecfps.new_zeros(1, n_bits+dupl_bits), frag_ecfps])      # padding feature is zero vector

    # train valid split
    train_data = Subset(dataset, df.loc[df.test==0].index.tolist())
    valid_data = Subset(dataset, df.loc[df.test==-1].index.tolist()) if validation & np.any(df.test==-1) else None

    # make data loader
    sampler = DistributedSampler(train_data, num_replicas= n_gpu, rank= rank, shuffle= True) if multigpu else None
    train_loader = DataLoader(train_data, batch_size= batch_size, shuffle= not multigpu,
                              sampler= sampler, pin_memory= True, collate_fn= collate_pad_fn)
    valid = bool(valid_data)
    if valid:
        valid_sampler = DistributedSampler(valid_data, num_replicas= n_gpu, rank= rank, shuffle= False) if multigpu else None
        valid_loader = DataLoader(valid_data, batch_size= batch_size, shuffle= False,
                                  sampler= valid_sampler, pin_memory= True, collate_fn= collate_pad_fn)

    if rank == 0:
        print(f'data: {data_path}', flush= True)
        print(f'train: {len(train_data)}, valid: {sum(df.test==-1)}, test: {sum(df.test==1)}, useChiral: {useChiral}, n_jobs: {n_jobs}', flush= True)
        print(f'fragments: {len(uni_fragments)}, feature: {frag_ecfps.shape[-1]}, tree: ({max_depth}, {max_degree}), cond: {len(pnames)}', flush= True)
        print(f'---{datetime.datetime.now()}: Loading data done. (elapsed time: {second2date(time.time()-s)})---\n', flush= True)

    # calculate loss weight
    num_labels = len(freq_label)
    freq_label = torch.tensor(freq_label)
    freq_label[freq_label > 1000] = 1000                              # limitation
    loss_weight_label = freq_label.max() / freq_label
    loss_weight_label[loss_weight_label == float('Inf')]  = 0.001     # padding weight

    # define model
    num_labels = frag_ecfps.shape[0]
    model = FTTVAE(num_labels, max_depth, max_degree, feat_dim, latent_dim,
                   d_model, d_ff, num_layers, num_heads, activation, dropout)
    model = CVAEwrapper(model, pnames, list(props.values())).to(device)
    if load_epoch:
        model.load_state_dict(torch.load(os.path.join(result_path, 'models', f'model_iter{load_epoch}.pth'), map_location= device))
        if rank==0: print(f'model loaded: {os.path.join(result_path, "models", f"model_iter{load_epoch}.pth")}', flush= True)

    if multigpu: 
        model = DDP(model, device_ids= [rank], find_unused_parameters= True)

    optimizer = optim.Adam(model.parameters(), lr= lr, eps= 1e-3)
    if load_epoch:
        state_dict = torch.load(os.path.join(result_path, 'models', f'optim_iter{load_epoch}.pth'), map_location= device)
        if state_dict['param_groups'][0]['lr'] == lr:
            optimizer.load_state_dict(state_dict)
            if rank==0: print(f'optimizer loaded: {os.path.join(result_path, "models", f"optim_iter{load_epoch}.pth")}', flush= True)
    
    # define loss
    loss_weight_label = loss_weight_label.to(device) if loss_weight_label is not None else None
    crirerion = nn.CrossEntropyLoss(weight= loss_weight_label)

    # release memory
    del df, df_frag, uni_fragments, freq_label, dataset, train_data, valid_data, loss_weight_label
    gc.collect()
    if multigpu: dist.barrier()


    ## train 
    if rank == 0:
        print(f'---{datetime.datetime.now()}: Training start (valid: {valid}).---', flush= True)
        print(f'epoch: {epochs}, batch_size: {batch_size}, learning_rate: {lr}, dropout: {dropout}, save: {save_epoch}, kl_w: {kl_w}, l_w: {l_w}', flush= True)

    # training
    metrics = ['total', 'kl', 'label', 'label_acc', 'label_pad_acc']
    TRAIN_LOSS, VALID_LOSS = [], []
    start = s = time.time()
    if rank==0:
        filename = os.path.join(result_path, 'train', f'train_{datetime.date.today()}.txt')
        with open(filename, 'a') as f:
            f.write(f'---{datetime.datetime.now()}: Training start.---\n')

    metric = args.select_metric      # metric for select best model. 0: total, 1: kl, 2: label loss, 3: label acc
    before = 0 if metric > 2 else float('inf')
    patient = 0
    iters = len(train_loader)
    epochs = load_epoch + epochs
    for epoch in range(load_epoch, epochs):
        model.train()
        if multigpu: sampler.set_epoch(epoch)
        train_losses = []
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            frag_indices = data[0]
            features = frag_ecfps[frag_indices.flatten()].reshape(frag_indices.shape[0], frag_indices.shape[1], -1).to(device)
            positions = data[1].to(device)
            conditions = {key: data[2][:, i].to(device) for i, key in enumerate(pnames)}
            target = torch.hstack([frag_indices.detach(), torch.zeros(frag_indices.shape[0], 1)]).flatten().long().to(device)

            # make mask
            nan_mask = torch.where(data[2].isnan(), 0, 1)
            frag_indices = torch.hstack([nan_mask, torch.full((frag_indices.shape[0], 1), -1), frag_indices.detach()]).to(device)  # for super root & condition
            src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(frag_indices, frag_indices, pad_idx= 0, batch_first= True)
            tgt_mask[:, :nan_mask.shape[-1]+1] = 0      # no sequence mask of conditions

            # forward
            z, mu, ln_var, output = model(features, positions, conditions,
                                          src_mask, src_pad_mask, 
                                          tgt_mask, tgt_pad_mask)     # output: shape= (B, L+1, num_labels)

            # calc loss
            kl_loss = batched_kl_divergence(mu, ln_var)
            label_loss = crirerion(input= output.view(-1, num_labels), target= target)

            # backward
            total_loss = kl_w * kl_loss + l_w * label_loss
            total_loss.backward()
            optimizer.step()

            # calc accuracy
            equals = output.argmax(dim= -1).flatten().eq(target)
            label_acc = equals[target!=0].sum() / (target!=0).sum()
            label_pad_acc = equals.sum() / target.shape[0]

            if multigpu:
                dist.all_reduce(total_loss, op= dist.ReduceOp.SUM)
                dist.all_reduce(kl_loss, op= dist.ReduceOp.SUM)
                dist.all_reduce(label_loss, op= dist.ReduceOp.SUM)
                dist.all_reduce(label_acc, op= dist.ReduceOp.SUM)
                dist.all_reduce(label_pad_acc, op= dist.ReduceOp.SUM)

            if rank==0:
                train_losses.append([total_loss.item()/n_gpu, kl_loss.item()/n_gpu, label_loss.item()/n_gpu, label_acc.item()/n_gpu, label_pad_acc.item()/n_gpu])
                # print(f'<{i+1:0=3}/{iters:0=3}> ' + ', '.join([f'{m}: {l:.4f}' for m, l in zip(metrics, train_losses[-1])]) + f', elapsed time: {second2date(time.time()-s)}', flush= True)

        # validation and model save
        if ((epoch+1) % save_epoch == 0) | ((epoch+1) == epochs):
            if multigpu: dist.barrier() 
            if valid:
                model.eval()
                valid_losses = []
                with torch.no_grad():
                    for data in valid_loader:
                        frag_indices = data[0]
                        features = frag_ecfps[frag_indices.flatten()].reshape(frag_indices.shape[0], frag_indices.shape[1], -1).to(device)
                        positions = data[1].to(device)
                        conditions = {key: data[2][:, i].float().to(device) for i, key in enumerate(pnames)}
                        target = torch.hstack([frag_indices.detach(), torch.zeros(frag_indices.shape[0], 1)]).flatten().long().to(device)

                        # make mask
                        nan_mask = torch.where(data[2].isnan(), 0, 1)
                        frag_indices = torch.hstack([nan_mask, torch.full((frag_indices.shape[0], 1), -1), frag_indices.detach()]).to(device)  # for super root & condition
                        src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(frag_indices, frag_indices, pad_idx= 0, batch_first= True)
                        tgt_mask[:, :nan_mask.shape[-1]+1] = 0      # no sequence mask of conditions

                        # forward
                        z, mu, ln_var, output = model(features, positions, conditions,
                                                      src_mask, src_pad_mask, 
                                                      tgt_mask, tgt_pad_mask, sequential= False)     # output: shape= (B, L+1, num_labels)

                        # calc loss
                        kl_loss = batched_kl_divergence(mu, ln_var)
                        label_loss = crirerion(input= output.view(-1, num_labels), target= target)
                        total_loss = kl_w * kl_loss + l_w * label_loss
                        
                        # calc accuracy
                        equals = output.argmax(dim= -1).flatten().eq(target)
                        label_acc = equals[target!=0].sum() / (target!=0).sum()
                        label_pad_acc = equals.sum() / target.shape[0]

                        if multigpu:
                            dist.all_reduce(total_loss, op= dist.ReduceOp.SUM)
                            dist.all_reduce(kl_loss, op= dist.ReduceOp.SUM)
                            dist.all_reduce(label_loss, op= dist.ReduceOp.SUM)
                            dist.all_reduce(label_acc, op= dist.ReduceOp.SUM)
                            dist.all_reduce(label_pad_acc, op= dist.ReduceOp.SUM)

                        if rank == 0:
                            valid_losses.append([total_loss.item()/n_gpu, kl_loss.item()/n_gpu, label_loss.item()/n_gpu, label_acc.item()/n_gpu, label_pad_acc.item()/n_gpu])
                if rank==0: 
                    VALID_LOSS.append([np.mean(losses) for losses in zip(*valid_losses)])
                    print(f'[{epoch+1:0=3}/{epochs:0=3}] <valid> ' + ', '.join([f'{m}: {l:.4f}' for m, l in zip(metrics, VALID_LOSS[-1])]) + f', elapsed time: {second2date(time.time()-s)}', flush= True)
            
            # save models
            if rank == 0:
                if multigpu:
                    torch.save(model.module.state_dict(), os.path.join(result_path, 'models', f'model_iter{epoch+1}.pth'))
                else:
                    torch.save(model.state_dict(), os.path.join(result_path, 'models', f'model_iter{epoch+1}.pth'))
                torch.save(optimizer.state_dict(), os.path.join(result_path, 'models', f'optim_iter{epoch+1}.pth'))
                print(f'model_iter{epoch+1} saved.', flush= True)

        # best model save
        if rank == 0:
            TRAIN_LOSS.append([np.mean(losses) for losses in zip(*train_losses)])
            # selected_metric = VALID_LOSS[-1][metric] if valid else TRAIN_LOSS[-1][metric]
            selected_metric = TRAIN_LOSS[-1][metric]
            tmp = selected_metric if (metric > 2) else -1 * selected_metric
            if tmp > before:
                ep = epoch + 1
                before = tmp
                best_state_dict = deepcopy(model.module.state_dict()) if multigpu else deepcopy(model.state_dict())
                patient = 0
            else:
                patient += 1

            with open(filename, 'a') as f:
                f.write(f'[{epoch+1:0=3}/{epochs:0=3}] <train> ' + ', '.join([f'{m}: {l:.4f}' for m, l in zip(metrics, TRAIN_LOSS[-1])]) + f', elapsed time: {second2date(time.time()-s)}\n')
            if (epoch == 0) | ((epoch+1) % 5 == 0):
                print(f'[{epoch+1:0=3}/{epochs:0=3}] <train> ' + ', '.join([f'{m}: {l:.4f}' for m, l in zip(metrics, TRAIN_LOSS[-1])]) + f', elapsed time: {second2date(time.time()-s)}', flush= True)

            if (patient > 5) | ((epoch+1) == epochs):
                torch.save(best_state_dict, os.path.join(result_path, 'models', f'model_best.pth'))
                print(f'model_iter{ep} saved. [{metrics[metric]}: {before:.4f}]', flush= True)
                patient = 0

        s = time.time()

    # save loss and reconstruction
    if rank==0:
        with open(filename, 'a') as f:
            f.write(f'---{datetime.datetime.now()}: Training done.---\n')
        TRAIN_LOSS = list(zip(*TRAIN_LOSS))
        df_loss = list2pdData(TRAIN_LOSS, metrics)
        df_loss.to_csv(os.path.join(result_path, 'train', f'loss{load_epoch}-{epochs}.csv'), index= False)
        if VALID_LOSS:
            VALID_LOSS = list(zip(*VALID_LOSS))
            df_loss = list2pdData(VALID_LOSS, metrics)
            df_loss.to_csv(os.path.join(result_path, 'valid', f'loss{load_epoch}-{epochs}.csv'), index= False)
        print(f'---{datetime.datetime.now()}: Training done. (elapsed time: {second2date(time.time()-start)})---\n', flush= True)



if __name__ == '__main__':
    yml_file = args.yml

    print(f'---{datetime.datetime.now()}: start.---', flush= True)

    ## check environments
    n_gpu = torch.cuda.device_count()
    print(f'GPU [{",".join([str(g) for g in args.gpus])}] is available: {torch.cuda.is_available()}', flush= True)
    if n_gpu > 1:
        print(f'DDP is available: {dist.is_available()}\n', flush= True)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = args.master_port
        os.environ['MKL_THREADING_LAYER'] = 'GNU'

    s = time.time()
    if n_gpu > 1:
        n_jobs = args.n_jobs // n_gpu if args.n_jobs > n_gpu else 1
        mp_args = (yml_file, args.load_epoch, args.save_interval, args.valid, n_jobs, args.seed)
        mp.spawn(train, nprocs= n_gpu, args= mp_args, join=True)
    else:
        train(0, yml_file, args.load_epoch, args.save_interval, args.valid, args.n_jobs, args.seed)
    
    print(f'---{datetime.datetime.now()}: all process done. (elapsed time: {second2date(time.time()-s)})---', flush= True)