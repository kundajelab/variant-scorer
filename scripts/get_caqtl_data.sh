#!/usr/bin/bash

# Selin Jessa
# July 2025

# Purpose: Documenting the process of setting up caQTL data for package testing. These data
# are provided with the ChromBPNet manuscript (Pampari et al, biorxiv, 2024).
# These variants provided are 1-based in hg38.

# 1. Download the caQTL data (https://www.synapse.org/Synapse:syn64126781)
# Requires synapse client & authentication first.
if [[ ! -f ../tests/data/raw/caqtls.african.lcls.benchmarking.all.tsv ]]; then
	echo "Downloading caQTL data from Synapse..."
	synapse get syn64126781
	gunzip caqtls.african.lcls.benchmarking.all.tsv.gz
	mv caqtls.african.lcls.benchmarking.all.tsv ../tests/data/raw/
else
	echo "caQTL data already downloaded."
fi

TEST_DIR="../tests/data"

# 2. Set rsids to keep
RSID="rs7417106|rs3121577|rs2465131|rs2488995|rs2488996|rs7527973|rs2063455|rs2685245|rs4402801|rs4854274"

# 3. Get appropriate columns and filter to a selected variants
# 1 var.chr : Chromosome of the variant (GRCh38)
# 2 var.pos_hg38 : Position of the variant (GRCh38, 1-based)
# 3 var.allele1 : Allele 1 for the variant
# 4 var.allele2 : Allele 2 for the variant
# 5 var.isused : True if variant is used in final ChromBPNet benchmarking
# 27 pred.chrombpnet.encsr637xsc.varscore.logfc : ChromBPNet logFC predictions in encid encsr637xsc
# 28 pred.chrombpnet.encsr637xsc.varscore.jsd : ChromBPNet JSD predictions in encid encsr637xsc
# 33 var.snp_id : variant identifier 1
# 36 var.dbsnp_rsid : dbSNP rsid identifier
cut -f 1,2,3,4,5,27,28,33,36 ${TEST_DIR}/raw/caqtls.african.lcls.benchmarking.all.tsv \
	| awk -F'\t' -v rsid_regex="$RSID" 'BEGIN{OFS="\t"} {if (NR==1 || $9 ~ rsid_regex) print $1,$2,$3,$4,$5,$6,$7,$8,$9}' \
	> ${TEST_DIR}/caqtls.african.lcls.benchmarking.subset.tsv

# 3. Convert to various input formats for testing

# BED (0-based: ['chr', 'pos', 'end', 'allele1', 'allele2', 'variant_id'])
awk 'BEGIN{OFS="\t"} {print $1,$2-1,$2,$3,$4,$9}' ${TEST_DIR}/caqtls.african.lcls.benchmarking.subset.tsv \
	| tail -n+2 \
	> ${TEST_DIR}/test.bed

# ChromBPNet input format (1-based, ['chr', 'pos', 'allele1', 'allele2', 'variant_id'])
awk 'BEGIN{OFS="\t"} {print $1,$2,$3,$4,$9}' ${TEST_DIR}/caqtls.african.lcls.benchmarking.subset.tsv \
	| tail -n+2 \
	> ${TEST_DIR}/test.chrombpnet.tsv

# ChromBPNet format missing 'chr' prefix
sed 's/^chr//g' ${TEST_DIR}/test.chrombpnet.tsv > ${TEST_DIR}/test.chrombpnet.no_chr.tsv

# plink format (1-based, ['chr', 'variant_id', 'ignore1', 'pos', 'allele1', 'allele2'])
awk 'BEGIN{OFS="\t"} {print $1,$9,"0",$2,$3,$4}' ${TEST_DIR}/caqtls.african.lcls.benchmarking.subset.tsv \
	| tail -n+2 \
	> ${TEST_DIR}/test.plink.tsv

# original format (1-based, ['chr', 'pos', 'variant_id', 'allele1', 'allele2'])
awk 'BEGIN{OFS="\t"} {print $1,$2,$9,$3,$4}' ${TEST_DIR}/caqtls.african.lcls.benchmarking.subset.tsv \
	| tail -n+2 \
	> ${TEST_DIR}/test.original.tsv