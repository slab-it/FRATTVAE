#!/bin/bash

source /yourdirectory/.pyenv/versions/anaconda3-2021.11/bin/activate 
conda activate py310chem

rd_filters filter --in your_path --prefix output_file --rules data/rules.json --alerts data/alert_collection.csv --np 24