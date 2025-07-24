#!/usr/bin/bash

# Selin Jessa
# July 2025

wd=`pwd`

# Purpose: Documenting the process of setting up caQTL data for package testing. These data
# are provided with the ChromBPNet manuscript (Pampari et al, biorxiv, 2024).
# These variants are 1-based in hg38.
mkdir -p ~/Downloads/variant-scorer/data
cd ~/Downloads/variant-scorer/data

# 1. Download the caQTL data (https://www.synapse.org/Synapse:syn64126781)
# Requires synapse client & authentication first.
synapse get syn64126781

# 2. Extract the files
tar -xvf caqtls.african.lcls.benchmarking.all.tsv.gz

# 3. Filter to isused=True and extract the appropriate columns
# var.chr : Chromosome of the variant (GRCh38)
# var.pos_hg38 : Position of the variant (GRCh38, 1-based)
# var.allele1 : Allele 1 for the variant
# var.allele2 : Allele 2 for the variant
# var.isused : True if variant is used in final ChromBPNet benchmarking
# pred.chrombpnet.encsr637xsc.varscore.logfc : ChromBPNet logFC predictions in encid encsr637xsc
# pred.chrombpnet.encsr637xsc.varscore.jsd : ChromBPNet JSD predictions in encid encsr637xsc
# pred.chrombpnet.encsr637xsc.varscore.ips : ChromBPNet IPS predictions in encid encsr637xsc
# var.dbsnp_rsid : dbSNP rsid identifier
# var.snp_id : variant identifier 1
# var.snp : variant identifier 2
cut -f 1,2,3,4,5,24,25,26,33,35,36 caqtls.african.lcls.benchmarking.all.tsv \
	| awk -F'\t' 'NR==1 || $5=="True"' \
	> caqtls.african.lcls.benchmarking.filtered.tsv

# Make a subset file with just 10 variants from chr1 and chr2
head -n 1 caqtls.african.lcls.benchmarking.filtered.tsv > caqtls.african.lcls.benchmarking.subset.tsv
grep chr1 caqtls.african.lcls.benchmarking.filtered.tsv | tail -n+2 | head -n 4 >> caqtls.african.lcls.benchmarking.subset.tsv
grep chr2 caqtls.african.lcls.benchmarking.filtered.tsv | head -n 2 >> caqtls.african.lcls.benchmarking.subset.tsv

# Now we have just 6 variants.
wc -l caqtls.african.lcls.benchmarking.subset.tsv

# 4. Move the file to the package data directory
cd $wd
mv ~/Downloads/variant-scorer/data/caqtls.african.lcls.benchmarking.subset.tsv ../tests/data

# 5. Convert to various input formats for testing

# BED (0-based: ['chr', 'pos', 'end', 'allele1', 'allele2', 'variant_id'])
awk 'BEGIN{OFS="\t"} {print $1,$2-1,$3,$4,$11}' caqtls.african.lcls.benchmarking.subset.tsv \
	| tail -n+2 \
	> test.bed

# ChromBPNet input format (1-based, ['chr', 'pos', 'allele1', 'allele2', 'variant_id'])
awk 'BEGIN{OFS="\t"} {print $1,$2,$4,$5,$11}' caqtls.african.lcls.benchmarking.subset.tsv \
	| tail -n+2 \
	> test.chrombpnet.tsv

# ChromBPNet format missing 'chr' prefix
sed 's/^chr//g' test.chrombpnet.tsv > test.chrombpnet.no_chr.tsv

# plink format (1-based, ['chr', 'variant_id', 'ignore1', 'pos', 'allele1', 'allele2'])
awk 'BEGIN{OFS="\t"} {print $1,$11,"0",$2,$4,$5}' caqtls.african.lcls.benchmarking.subset.tsv \
	| tail -n+2 \
	> test.plink.tsv

# original format (1-based, ['chr', 'pos', 'variant_id', 'allele1', 'allele2'])
awk 'BEGIN{OFS="\t"} {print $1,$2,$11,$4,$5}' caqtls.african.lcls.benchmarking.subset.tsv \
	| tail -n+2 \
	> test.original.tsv
