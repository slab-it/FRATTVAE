#!/bin/bash

source yourenviroment     # activate your virtual enviroment ex.conda, venv

pip3 install --upgrade pip setuptools
pip3 install wheel
pip3 install jupyter pipdeptree \
                     numpy \
                     pandas==1.5.2 \
                     pyyaml \
                     joblib \
                     matplotlib \
                     ordered-set \
                     umap-learn \
                     scikit-learn==1.1.3 \
                     rdkit==2023.3.1 \
                     timeout-decorator \
                     scipy==1.11.3 \
                     molvs
pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install dgl==1.1.1+cu113 -f https://data.dgl.ai/wheels/cu113/repo.html
pip3 install fcd-torch guacamol

deactivate