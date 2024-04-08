#!/bin/bash
# run multiple jobs using JBSUB

# use the following code for debugging in an interactive session:
# >> jbsub -cores 1+1 -q x86_6h -mem 15g -interactive bash
# >> nohup python wrapper.py --fname_out_root myson_top_n_perc_0.5 --top_n_perc 0.5 --epochs 10 > nohup.out 2>&1 &

queue="x86_6h"
# FIRST STEP LAST
epochs=($(seq 1 1 5))
mem="100g"
batches=(50000)
# numbers=(0.5 1.0 5.0 10.0 15.0 20.0)
numbers=(0.5)
name="CMCL"

seq_data="/dccstor/ukb-pgx/comical/comical/data/snp-encodings-from-vcf.csv"
idp_data="/dccstor/ukb-pgx/comical/comical/data/T1_struct_brainMRI_IDPs.csv"
idp_map="/dccstor/ukb-pgx/comical/comical/data/T1mri.csv"
idp_bucket="/dccstor/ukb-pgx/comical/comical/data/IDPs_and_disease_mapping.csv"
seq_bucket="/dccstor/ukb-pgx/comical/comical/data/SNPs_and_disease_mapping_with_pvalues.csv"

# adding '-require a100_80gb' never started to run...so resource never became available

for batch_size in "${batches[@]}"; do-

	for ep in "${epochs[@]}"; do

		for num in "${numbers[@]}"; do

			jobname=${name}"_per_"${num}"_ep_"${ep}"_ba_"${batch_size}

    		jbsub -cores 2+2 -q $queue -mem $mem -name $jobname \
	 			  -out ${jobname}.out -err ${jobname}.err python wrapper.py --fname_out_root $jobname \
	    		  --top_n_perc $num --epochs $ep --batch_size $batch_size \
	    		  -ps $seq_data -pi $idp_data -pm $idp_map -pbi $idp_bucket -pbs $seq_bucket
		done

	done

done

