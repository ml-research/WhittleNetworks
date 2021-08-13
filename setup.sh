#!/bin/bash

rm -rf venv_wnet/
virtualenv --system-site-packages -p python3.6 ./venv_wnet
source ./venv_wnet/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
