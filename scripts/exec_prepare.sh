#!/bin/bash

export PYTHONPATH="/yourdirectory/scripts:$PYTHONPATH"
export DGLBACKEND="pytorch"

source yourenviroment

data_path="/yourdirectory/data/example_standardized.csv"

echo date >> prepare.log

# basic
python preparation.py $data_path \
                      --seed 0 \
                      --maxLength 32 \
                      --maxDegree 16 \
                      --minSize 1 \
                      --epoch 1000 \
                      --batch_size 2048 \
                      --lr 0.0001 \
                      --kl_w 0.0005 \
                      --l_w 2.0 >> prepare.log 

# # kl-annealing
# python preparation.py $data_path \
#                       --seed 0 \
#                       --maxLength 32 \
#                       --maxDegree 16 \
#                       --minSize 1 \
#                       --epoch 1000 \
#                       --batch_size 512 \
#                       --lr 0.0001 \
#                       --kl_w 0.0005 \
#                       --anneal_epoch 500:10 \
#                       --l_w 2.0 >> prepare.log

# # conditional
# python preparation.py $data_path \
#                       --seed 0 \
#                       --maxLength 32 \
#                       --maxDegree 16 \
#                       --minSize 1 \
#                       --epoch 1000 \
#                       --batch_size 1024 \
#                       --condition MW:1 \
#                       --condition logP:1 \
#                       --condition QED:1 \
#                       --condition SA:1 \
#                       --condition NP:1 \
#                       --condition TPSA:1 \
#                       --lr 0.0001 \
#                       --kl_w 0.0005 \
#                       --l_w 2.0 >> prepare.log