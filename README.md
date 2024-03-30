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

## Precedure of Training and Generation
### 1. Preprocessing
### (1.0.) 　Standardize SMILES
To canonicalize and sanitize SMILES, run `exec_standardize.sh` only once. your data must be in csv format and have a column named 'SMILES'. If there is not a column called 'test' in your data, it will be split into train/valid/test data sets (0: train, 1: test, -1: valid). The standardized data is saved as `*_standardized.csv`.

exec_standardize.sh:
```
#!/bin/bash

export PYTHONPATH="/yourdirectory/scripts:$PYTHONPATH"
export DGLBACKEND="pytorch"

source yourenviroment
data_path="/yourdirectory/data/example.csv"

python utils/standardize_smiles.py $data_path --n_jobs 16 >> prepare.log
```
Please change 'yourdirectory' and 'yourenviroment' to the correct paths.

### 1.1. 　Setting Hyperparameters and Save Directory
Create `savedir` named `dataname_{taskname}_{MMDD}` in `/yourdirectory/results/.`, and `params.yml` which is hyperparameters list in `/savedir/input_data/.`.
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
