# Fragment Tree Transoformer-based VAE (FTTVAE)

This repository contains training and generation code for FTTVAE. FTTVAE can handle large amount of varius compounds ranging from 'Drug-like' to 'Natural'. In addition, the latent space constructed by FTTVAE is useful to molecular generation and optimization.

## Requirements
* Python==3.10.8
* Scipy==1.11.3
* Pytorch>=1.12.1 (We only testd 1.12.1 and 2.0.1)
* DGL>=1.1.1
* RDkit==2023.2.1
* molvs (https://github.com/mcs07/MolVS)
* moses (https://github.com/molecularsets/moses)
* guacamol (https://github.com/BenevolentAI/guacamol)

To install these packages, follow the respective instructions.

## Quick Start
Essential packages can be installed via `pip`, but the version of CUDA is up to you (Default: 11.3). 
Please execute `enviroment.sh` in your python virtual enviroment.
```
sh enviroment.sh
```
If you use Docker, you can use the Dockerfile to build your environment.
```
docker build . --network=host -t <IMAGE NAME>
docker run -itd --runtime=nvidia --shn-size 32g -t <CONTAINER NAME> <IMAGE ID>
```

## 0. Preparation
### (0.0.) 　Standardize SMILES
To canonicalize and sanitize SMILES, run `exec_standardize.sh` only once. your data must be in csv format and have a column named 'SMILES'. If there is not a column called 'test' in your data, it will be split into train/valid/test data sets (0: train, 1: test, -1: valid). The standardized data is saved as `*_standardized.csv`.

exec_standardize.sh:
```
#!/bin/bash

export PYTHONPATH="/yourdirectory/scripts:$PYTHONPATH"
export DGLBACKEND="pytorch"

source yourenviroment
data_path="/yourdirectory/data/example.csv"

python utils/standardize_smiles.py $data_path --n_jobs 24 >> prepare.log
```
* --n_jobs: Number of cpu workers.

Please change 'yourdirectory' and 'yourenviroment' to the correct paths.

### 0.1. 　Setting Hyperparameters and Save Directory
```
python preparation.py "/yourdirectory/data/example_standardized.csv" \
                      --seed 0 \
                      --maxLength 32 \
                      --maxDegree 16 \
                      --minSize 1 \
                      --epoch 1000 \
                      --batch_size 2048 \
                      --lr 0.0001 \
                      --kl_w 0.0005 \
                      --l_w 2.0 >> prepare.log 
```
Create `savedir` named `dataname_{taskname}_{MMDD}` in `/yourdirectory/results/.`, and `params.yml` which is hyperparameters list in `/savedir/input_data/.`.

## 1. Precedure of Training and Generation
Please refer to `exec_vae.sh`.

### 1.1. Preprocessing
```
path="/yourdirectory/results/examples_standardized_struct_{MMDD}"
ymlFile=$path'/input_data/params.yml'
python preprocessing.py ${ymlfile} --njobs 24 >> $path'/preprocess.log'
```
* --ymlfile: the path of `params.yml`.
* --n_jobs: Number of cpu workers.

After execution, `fragments.csv` and `dataset.pkl` are created in `/savedir/input_data/.`

### 1.2. Training
```
python train.py ${ymlFile} --gpus 0 1 2 3 --n_jobs 24 --load_epoch $load_epoch --valid > $path'/train'$load_epoch'.log'
```
* --gpus: IDs of GPU. If multiple GPUs are given, they are used for DDP training.
* --n_jobs: Number of cpu workers.
* --load_epoch: load `$load_epoch`-epoch trained model. Use to resume learning from any epoch.
* --valid: To Validate or Not to Validate.

After execution, the model checkepoint is saved as `model_best.pth` in `/savedir/model/.`

### 1.3. Reconstruntion and Generation
Caluculate reconstruction accuracy and MOSES+GuacaMol metrics.
```
python test.py ${ymlFile} --gpu 0 --k 10000 --N 5 --n_jobs 24 > $path'/test.log'
```
* --gpu: ID of GPU. multi GPUs are not supported.
* --k: Number of moldecules generated.
* --N: Iteration number of generation.
* --n_jobs: Number of cpu workers.
* --gen: Set if you only want Generation.

After execution, the results of reconstruction and generation are saved in `/savedir/test/.` and `/savedir/generate/.` respectively.

## Pretrained Model
In the directory `results`, several trained model are exist. You can generate molecules using the trained model.
\
ex. GuacaMol
```
#!/bin/bash

export PYTHONPATH="/yourdirectory/scripts:$PYTHONPATH"
export DGLBACKEND="pytorch"

source yourenviroment

path="/yourdirectory/results/GuacaMol_standardized_struct_0216"
ymlFile=$path'/input_data/params.yml'
load_epoch=0

nohup python3 test.py ${ymlFile} --N 5 --k 10000 --gpu 0 --n_jobs 24 --gen > $path'/test.log' &
```
