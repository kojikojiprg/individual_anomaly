#!/usr/bin/env bash
pip install -U pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu115

git submodule update --init

# installation for mmdet
cd submodules/mmdet
pip install -r requirements.txt
pip install -v -e .
cd ../../

# installation for mmpose
cd submodules/mmpose
pip install -r requirements.txt
pip install -v -e .
cd ../../

# install unitrack
cd submodules/unitrack
python setup.py
# sklearn >= 0.23 changed this function name
sed -i '/jaccard_similarity_score/d' utils/mask.py
cd ../../  # go back root of the project
