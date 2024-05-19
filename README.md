# Fragment Tree Transformer-based VAE (FRATTVAE)

This repository contains training and generation code for FRATTVAE. FRATTVAE can handle large amount of varius compounds ranging from 'Drug-like' to 'Natural'. In addition, the latent space constructed by FRATTVAE is useful to molecular generation and optimization.
FRATTVAE is implemented in this papar.

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
* `--n_jobs`: Number of cpu workers.

Please change 'yourdirectory' and 'yourenviroment' to the correct paths.

You can download several standardized datasets [here](https://drive.google.com/drive/folders/16LAR-wDdsNEAYbVT8KcG_DJtm6a7GhVP?usp=sharing) (ZINC250K, MOSES, GuacaMol, Polymer, NaturalProducts).

### 0.1. 　Setting Hyperparameters and Directory to save results
exec_prepare.sh:
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
After execution, `savedir` named `dataname_{taskname}_{MMDD}` in `/yourdirectory/results/.`, and `params.yml` which is hyperparameters list in `/savedir/input_data/.` are created.

Please adjust hyperparameters (batch_size, maxLength and so on) to match your datasets and GPU capacities.
For molecules with molecular weights greater than 500, it is recommended that maxLength be 32 or 64.

## 1. Precedure of Training and Generation
Please refer to `exec_vae.sh`.

### 1.1. Preprocessing
```
path="/yourdirectory/results/examples_standardized_struct_{MMDD}"
ymlFile=$path'/input_data/params.yml'
python preprocessing.py ${ymlfile} --njobs 24 >> $path'/preprocess.log'
```
* `--ymlfile`: the path of `params.yml`.
* `--n_jobs`: Number of cpu workers.

After execution, `fragments.csv` and `dataset.pkl` are created in `/savedir/input_data/.`

### 1.2. Training
```
python train.py ${ymlFile} --gpus 0 1 2 3 --n_jobs 24 --load_epoch $load_epoch --valid > $path'/train'$load_epoch'.log'
```
* `--gpus`: IDs of GPU. If multiple GPUs are given, they are used for DDP training.
* `--n_jobs`: Number of cpu workers.
* `--load_epoch`: load `$load_epoch`-epoch trained model. Use to resume learning from any epoch.
* `--valid`: To Validate or Not to Validate.

After execution, the model checkepoint is saved as `model_best.pth` in `/savedir/model/.`

### 1.3. Reconstruntion and Generation
Caluculate reconstruction accuracy and MOSES+GuacaMol metrics.
```
python test.py ${ymlFile} --gpu 0 --k 10000 --N 5 --n_jobs 24 > $path'/test.log'
```
* `--gpu`: ID of GPU. multi GPUs are not supported.
* `--k`: Number of moldecules generated.
* `--N`: Iteration number of generation.
* `--n_jobs`: Number of cpu workers.
* `--gen`: Set if you only want Generation.

After execution, the results of reconstruction and generation are saved in `/savedir/test/.` and `/savedir/generate/.` respectively.

## Conditional VAE
You can also train FRATTVAE with some conditions(logP, QED, SA, ...).
Condition values must be included in the datafile.
Conditions are selected as arguments in `preparation.py`. (See `exec_prepare.sh`)
If the condition value is a continuous value, enter condition key and value '1' (ex. MW:1).
If the condition value is a discrete value, enter condition key and value 'number of categories'.
```
python preparation.py $data_path \
                       --seed 0 \
                       --maxLength 32 \
                       --maxDegree 16 \
                       --minSize 1 \
                       --epoch 1000 \
                       --batch_size 1024 \
                       --condition MW:1 \
                       --condition logP:1 \
                       --lr 0.0001 \
                       --kl_w 0.0005 \
                       --l_w 2.0 >> prepare.log
```
You can exececute conditional training and generation using `cvae/exec_cvae.sh`

## Pretrained Model
You can generate molecules using trained models.
Download result directories containing trained models [here](https://drive.google.com/drive/folders/1VF7lFOlBUr6T5_ESnaj2xbs3hQnV5knz?usp=sharing) and unzip downloaded files. 
Next, please replace `yourdirectory` to your directories and rewrite `batch_size` to match gpu capacity　in `input_data/params.yml`.
\
ex. GuacaMol
```
#!/bin/bash

export PYTHONPATH="/yourdirectory/scripts:$PYTHONPATH"
export DGLBACKEND="pytorch"

source yourenviroment

path="/yourdirectory/results/GuacaMol_standardized_struct"
ymlFile=$path'/input_data/params.yml'

nohup python3 test.py ${ymlFile} --N 5 --k 10000 --gpu 0 --n_jobs 24 --gen > $path'/generate.log' &
```

## License

This software is released under a custom license.

Academic use of this software is free and does not require any permission.
We encourage academic users to cite our research paper (if applicable).

For commercial use, please contact the author for permission at [inukai10@dna.bio.keio.ac.jp].
