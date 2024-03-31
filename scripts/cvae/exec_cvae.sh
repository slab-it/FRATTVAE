#!/bin/bash

export PYTHONPATH="/yourdirectory/scripts:$PYTHONPATH"
export DGLBACKEND="pytorch"

module load python/3.11 cuda/11.8 cudnn/8.6 nccl/2.16 hpcx-mt/2.12
source /yourdirectory/pytorch+horovod/bin/activate

path="/yourdirectory/results/ZINC_JTVAE_standardized_condition_MW_logP_QED_SA_NP_TPSA_max16_useChiral1_0223"
ymlFile=$path'/input_data/params.yml'
load_epoch=0

# nohup python ../preprocessing.py ${ymlFile} --n_jobs 24 --normalize >> $path'/preprocess.log' &
# nohup python train.py ${ymlFile} --gpus 1 7 --n_jobs 24 --save_interval 50 --load_epoch $load_epoch --valid --master_port 12356 > $path'/train'$load_epoch'.log' &&
# nohup python test.py ${ymlFile} --N 5 --k 10000 --gpu 7 --n_jobs 24 > $path'/test.log' &

# nohup python test.py ${ymlFile} --N 5 --k 10000 --gpu 7 --n_jobs 24 --load_epoch $load_epoch  > $path'/test'$load_epoch'.log' &
# nohup python test.py ${ymlFile} --N 5 --k 10000 --gpu 7 --n_jobs 24 --load_epoch $load_epoch --gen > $path'/generate'$load_epoch'.log' &

conditions="MW"
values="800"
nohup python conditional_generation.py ${ymlFile} --keys $conditions --values $values --N 1 --k 10000 --gpu 0 --n_jobs 16 --load_epoch $load_epoch >> $path'/cond_generate'$load_epoch'.log' &
# nohup python improvement.py ${ymlFile} ../../data/jtvae_qed.txt --keys $conditions --values $values --gpu 7 --n_jobs 8 --load_epoch $load_epoch > $path'/improve'$load_epoch'.log' &
# nohup python constrained_improve.py ${ymlFile} ../../data/jtvae_qed.txt QED 1 --gpu 7 --n_jobs 8 --load_epoch $load_epoch > $path'/const_improve'$load_epoch'.log' &