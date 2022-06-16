#!/usr/bin/env bash

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu115

git submodule update --init

# installation for mmpose
cd submodules/mmpose
pip install -v -e .
cd ../../
