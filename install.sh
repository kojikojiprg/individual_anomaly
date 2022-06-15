#!/bin/sh
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu115

git submodule update --init

# installation for mmpose
mim install mmcv-full
cd submodule/mmpose
pip install -v -e .
cd ../../
