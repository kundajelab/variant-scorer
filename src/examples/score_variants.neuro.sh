#!/bin/bash

CUDA_VISIBLE_DEVICES=1

list=/oak/stanford/groups/akundaje/soumyak/refs/plink_1kg_hg38/all.1000G.EUR.QC.bim
genome=/oak/stanford/groups/akundaje/refs/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta
model=/srv/scratch/soumyak/neuro-variants/outs/1_31_2022_adpd_model_training/full_models/cluster1_fold0/chrombpnet_wo_bias.h5
out_prefix=/oak/stanford/groups/akundaje/projects/variant-scorer/examples/neuro.cluster1_fold0.all.1000G.EUR

time python ../variant_scoring.py -l $list \
                             -g $genome \
                             -m $model \
                             -o $out_prefix \
                             -sc plink \
                             -dm

