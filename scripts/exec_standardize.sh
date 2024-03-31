#!/bin/bash

export PYTHONPATH="/yourdirectory/scripts:$PYTHONPATH"
export DGLBACKEND="pytorch"

source yourenviroment

data_path="/yourdirectory/data/example.csv"

# Only the first time
python utils/standardize_smiles.py $data_path --n_jobs 16 >> prepare.log
