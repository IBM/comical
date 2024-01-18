#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh

conda activate /u/dmr/ukb-pgx/comical/comical/comical

python wrapper.py -fo testing_coco_new_testing -bz 256 -gpu 0 -tr_coco 1