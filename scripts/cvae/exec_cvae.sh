#!/bin/bash

export PYTHONPATH="/yourdirectory/scripts:$PYTHONPATH"
export DGLBACKEND="pytorch"

source yourenviroment

path="/yourdirectory/results/<save directory>"
ymlFile=$path'/input_data/params.yml'
load_epoch=0

nohup python ../preprocessing.py ${ymlFile} --n_jobs 24 --normalize >> $path'/preprocess.log' &
nohup python train.py ${ymlFile} --gpus 0 1 2 3 --n_jobs 24 --save_interval 50 --load_epoch $load_epoch --valid > $path'/train'$load_epoch'.log' &&
nohup python test.py ${ymlFile} --N 5 --k 10000 --gpu 7 --n_jobs 24 > $path'/test.log' &


## conditional generation
# conditions="MW"
# values="800"
# nohup python conditional_generation.py ${ymlFile} --keys $conditions --values $values --N 1 --k 10000 --gpu 0 --n_jobs 16 --load_epoch $load_epoch >> $path'/cond_generate'$load_epoch'.log' &
# nohup python improvement.py ${ymlFile} ../../data/jtvae_qed.txt --keys $conditions --values $values --gpu 7 --n_jobs 8 --load_epoch $load_epoch > $path'/improve'$load_epoch'.log' &
# nohup python constrained_improve.py ${ymlFile} ../../data/jtvae_qed.txt QED 1 --gpu 7 --n_jobs 8 --load_epoch $load_epoch > $path'/const_improve'$load_epoch'.log' &
