#!/bin/bash

set -e
set -u
set -o pipefail
set -x

#rm -rf output
mkdir -p output/

python -u ../src/variant_scoring.per_chrom.py \
  -l /oak/stanford/groups/akundaje/airanman/test_data/variant-scorer/shared/encode_variants.subset.tsv \
  -g /oak/stanford/groups/akundaje/airanman/test_data/variant-scorer/shared/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
  -s /oak/stanford/groups/akundaje/airanman/test_data/variant-scorer/shared/GRCh38_EBV.chrom.sizes.tsv \
  -m /oak/stanford/groups/akundaje/airanman/test_data/variant-scorer/ENCSR999NKW/fold_0/chrombpnet_wo_bias.h5 \
  -p /oak/stanford/groups/akundaje/airanman/test_data/variant-scorer/ENCSR999NKW/peaks.subset.bed.gz \
  -o output/test \
  -t 20 \
  -sc chrombpnet

