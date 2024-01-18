#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh

conda activate /u/dmr/ukb-pgx/comical/comical/comical

python wrapper.py -fo comical_new_top10snps_pairs -bz 7500 -gpu 0 -tr_coco 0 -e 10