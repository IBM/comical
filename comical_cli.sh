#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh

conda activate comical-env

python wrapper.py -fo comical_demo_run -bz 32768 -gpu 7 -e 10