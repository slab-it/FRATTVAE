#!/bin/bash

export PYTHONPATH="/yourdirectory/scripts:$PYTHONPATH"
export DGLBACKEND="pytorch"

source yourenviroment

path="/yourdirectory/results/<save directory>"
ymlFile=$path'/input_data/params.yml'
load_epoch=0

nohup python3 preprocessing.py ${ymlFile} --n_jobs 16 >> $path'/preprocess.log' &
nohup python3 train.py ${ymlFile} --gpus 0 1 2 3 --n_jobs 24 --save_interval 50 --load_epoch $load_epoch --valid --master_port 12355 > $path'/train'$load_epoch'.log' &&
nohup python3 test.py ${ymlFile} --N 5 --k 10000 --gpu 0 --n_jobs 24 > $path'/test.log' &

# nohup python test.py ${ymlFile} --N 5 --k 10000 --gpu 7 --n_jobs 24 --load_epoch $load_epoch  > $path'/test'$load_epoch'.log' &
# nohup python test.py ${ymlFile} --N 5 --k 10000 --gpu 7 --n_jobs 24 --max_nfrags 12 --load_epoch $load_epoch --gen > $path'/generate'$load_epoch'.log' &
