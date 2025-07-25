#!/bin/bash

python ../src/variant_annotation.py -p /oak/stanford/groups/akundaje/projects/chromatin-atlas-2022/ATAC/ENCSR611BQR/preprocessing/downloads/peaks.bed.gz -ge /oak/stanford/groups/akundaje/soumyak/refs/hg38/hg38.tss.bed -o annotations/test -sc chrombpnet -l /oak/stanford/groups/akundaje/airanman/nautilus-sync/gregor-luria/pvc/outputs/gregor-luria/variant_summary/ENCSR611BQR/.mean.variant_scores.tsv

