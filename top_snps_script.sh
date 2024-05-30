#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh

conda activate comical-env

top_n_percents=("0.01" "0.05" "0.1" "0.15" "0.2")
gpu_ids=(2 3 4 5 6)

export top_n_percents
export gpu_ids

parallel -j 5 --link --env top_n_percents --env gpu_ids --lb \
    'python wrapper.py -fo comical_top_{1}_percent --path_res results_top_n_percent -gpu {2}' ::: "${top_n_percents[@]}" ::: "${gpu_ids[@]}"
