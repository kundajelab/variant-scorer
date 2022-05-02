#!/bin/bash

set -e
set -o pipefail
set -u

list=/oak/stanford/groups/akundaje/soumyak/refs/plink_1kg_hg38/all.1000G.EUR.QC.bim                                                    genome=/oak/stanford/groups/akundaje/refs/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta
model=/oak/stanford/groups/akundaje/projects/neuro-variants/outs/1_31_2022_adpd_model_training/full_models/cluster1_fold0/chrombpnet_wo_bias.h5
bias=/oak/stanford/groups/akundaje/projects/neuro-variants/outs/1_31_2022_adpd_model_training/full_models/cluster1_fold0/chrombpnet.h5
out_prefix=/oak/stanford/groups/akundaje/projects/variant-scorer/examples/neuro.cluster1_fold0.all.1000G.EUR

JOBSCRIPT=/home/groups/akundaje/soumyak/variant-scorer/examples/jobscript.sh
job=neuro.cluster1_fold0

sbatch -J $job \
       -t 60 -c 2 --mem=20G -p akundaje,gpu --gpus 1 \
       -o /oak/stanford/groups/akundaje/projects/variant-scorer/examples/$job.log.txt \
       -e /oak/stanford/groups/akundaje/projects/variant-scorer/examples/$job.err.txt \
       $JOBSCRIPT /home/groups/akundaje/soumyak/variant-scorer/src/variant_scoring.py \
         -l $list \
         -g $genome \
         -m $model \
         -o $out_prefix \
         -sc plink \
         -b $bias \
         -dm

