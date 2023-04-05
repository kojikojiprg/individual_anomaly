#!/usr/bin/env bash
pip install -U pip
# pip install wheel  # deleted for avoiding mmcv-full error
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu115

git submodule update --init

# install mmcv
# cd submodules/mmcv
# pip install -r requirements.txt
# MMCV_WITH_OPS=1 pip install -e .
# cd ../../
# pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu115/torch1.12.1/index.html

# install mmdet
# cd submodules/mmdet
# pip install -r requirements.txt
# pip install -v -e .
# cd ../../

# install mmpose
# cd submodules/mmpose
# pip install -r requirements.txt
# pip install -v -e .
# cd ../../

# install unitrack
cd submodules/unitrack
pip install imageio  # install one by one because of avoiding error
pip install lap
pip install cython
pip install cython_bbox
python setup.py
sed -i '/jaccard_similarity_score/d' utils/mask.py # sklearn >= 0.23 changed this function name
cd ../../  # go back root of the project
