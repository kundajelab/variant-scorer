#!/bin/bash

CUDA_VISIBLE_DEVICES=1

list=/oak/stanford/groups/akundaje/soumyak/refs/plink_1kg_hg38/all.1000G.EUR.QC.bim
genome=/oak/stanford/groups/akundaje/refs/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta
model=/mnt/lab_data3/soumyak/CRC_finemap/output/chrombpnet/20220124_celltype_models_5foldCV_Myo2bias/Cancer_Associated_Fibroblasts/fold0/Cancer_Associated_Fibroblasts.h5
bias=/mnt/lab_data3/soumyak/CRC_finemap/output/chrombpnet/bias_models/Myofibroblasts_2/Myofibroblasts_2.bias.2114.1000.h5
out_prefix=/mnt/lab_data3/soumyak/variant-scorer/examples/crc.caf_fold0.all.1000G.EUR

time python ../src/variant_scoring.py -l $list \
                             -g $genome \
                             -m $model \
                             -o $out_prefix \
                             -sc plink \
                             -b $bias \
                             -li \
                             -dm

