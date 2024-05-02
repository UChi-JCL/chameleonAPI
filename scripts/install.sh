#!/usr/bin/env bash
miniconda_path=$(dirname $(dirname $(which conda)))

# Echoing the Miniconda path
echo "Miniconda is installed at: $miniconda_path"
source ${miniconda_path}/etc/profile.d/conda.sh
conda env create -f env.yml 
conda activate cc
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e nlp/transformers
git clone https://github.com/mapillary/inplace_abn.git
cd inplace_abn
python setup.py install