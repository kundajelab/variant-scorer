import pandas as pd
import numpy as np
import os
import subprocess

from utils.argmanager import *
from utils.helpers import *


DEFAULT_CLOSEST_GENE_COUNT = 3

def main(args = None):
    if args is None:
        args = fetch_variant_annotation_args()
    variant_scores_file = args.list
    output_prefix = args.out_prefix
    peak_path = args.peaks
    tss_path = args.closest_genes

    variant_scores = pd.read_table(variant_scores_file)
    tmp_bed_file_path = output_prefix + ".variant_table.tmp.bed"

    if args.schema == "bed":
        if variant_scores['pos'].equals(variant_scores['end']):
            variant_scores['pos'] = variant_scores['pos'] - 1
        variant_scores_bed_format = variant_scores[['chr','pos','end','allele1','allele2','variant_id']].copy()
    else:
        ### convert to bed format
        variant_scores_bed_format = variant_scores[['chr','pos','allele1','allele2','variant_id']].copy()
        variant_scores_bed_format['pos']  = variant_scores_bed_format.apply(lambda x: int(x.pos)-1, axis = 1)
        variant_scores_bed_format['end']  = variant_scores_bed_format.apply(lambda x: int(x.pos)+len(x.allele1), axis = 1)
        variant_scores_bed_format = variant_scores_bed_format[['chr','pos','end','allele1','allele2','variant_id']]
        variant_scores_bed_format = variant_scores_bed_format.sort_values(["chr","pos","end"])

    variant_scores_bed_format.to_csv(tmp_bed_file_path,\
                                     sep="\t",\
                                     header=None,\
                                     index=False)


    if args.closest_genes:

        print("annotating with closest genes")
        closest_gene_count = args.closest_gene_count if args.closest_gene_count else DEFAULT_CLOSEST_GENE_COUNT
        closest_gene_path = "%s.closest_genes.bed"%output_prefix
        gene_bedtools_intersect_cmd = f"bedtools closest -d -t first -k {closest_gene_count} -a {tmp_bed_file_path} -b {tss_path} > {closest_gene_path}"
        _ = subprocess.call(gene_bedtools_intersect_cmd,\
                            shell=True)

        closest_gene_df = pd.read_table(closest_gene_path, header=None)
        os.remove(closest_gene_path)

        print()
        print(closest_gene_df.head())
        print("Closest genes table shape:", closest_gene_df.shape)
        print()

        closest_genes = {}
        gene_dists = {}

        for index,row in closest_gene_df.iterrows():
            if not row[5] in closest_genes:
                closest_genes[row[5]] = []
                gene_dists[row[5]] = []
            closest_genes[row[5]].append(row[9])
            gene_dists[row[5]].append(row[10])

        closest_gene_df = closest_gene_df.rename({5:'variant_id'},axis=1)
        closest_gene_df = closest_gene_df[['variant_id']]

        for i in range(closest_gene_count):
            closest_gene_df[f'closest_gene_{i+1}'] = closest_gene_df['variant_id'].apply(lambda x: closest_genes[x][i] if len(closest_genes[x]) > i else '.')
            closest_gene_df[f'gene_distance_{i+1}'] = closest_gene_df['variant_id'].apply(lambda x: gene_dists[x][i] if len(closest_genes[x]) > i else '.')

        closest_gene_df.drop_duplicates(inplace=True)
        variant_scores = variant_scores.merge(closest_gene_df, on='variant_id', how='left')

    if args.peaks:

        print("annotating with peak overlap")
        peak_intersect_path = "%s.peak_overlap.bed"%output_prefix
        peak_bedtools_intersect_cmd = "bedtools intersect -wa -u -a %s -b %s > %s"%(tmp_bed_file_path, peak_path, peak_intersect_path)
        _ = subprocess.call(peak_bedtools_intersect_cmd,\
                            shell=True)

        peak_intersect_df = pd.read_table(peak_intersect_path, header=None)
        os.remove(peak_intersect_path)

        print()
        print(peak_intersect_df.head())
        print("Peak overlap table shape:", peak_intersect_df.shape)
        print()

        variant_scores['peak_overlap'] = variant_scores['variant_id'].isin(peak_intersect_df[5].tolist())

    if args.r2:
        r2_ld_filepath = args.r2

        r2_tsv_filepath = "/tmp/r2.tsv"
        with open(r2_ld_filepath, 'r') as r2_ld_file, open(r2_tsv_filepath, mode='w') as r2_tsv_file:
            # temp=r2_tsv_file.name
            for line in r2_ld_file:
                # Process the line
                line = '\t'.join(line.split())
                # Write the processed line to the output file, no need to specify end='' as '\n' is added explicitly
                r2_tsv_file.write(line + '\n')
            r2_tsv_file.flush()
            
        with open(r2_tsv_filepath, 'r') as r2_tsv_file:
            plink_variants = pd.read_table(r2_tsv_file)
            print("plink_variants:")
            print(plink_variants.head())
            print(plink_variants.shape)

            # Get just the lead variants, which is provided by the user.
            lead_variants = variant_scores[['chr', 'pos', 'variant_id']].copy()
            lead_variants['r2'] = 1.0
            lead_variants['lead_variant'] = lead_variants['variant_id']
            print("lead_variants:")
            print(lead_variants.head())
            print(lead_variants.shape)

            # Get just the ld variants.
            plink_ld_variants = plink_variants[['SNP_A','CHR_B','BP_B','SNP_B','R2']].copy()
            plink_ld_variants.columns = ['lead_variant', 'chr', 'pos', 'variant_id', 'r2']
            plink_ld_variants = plink_ld_variants[['chr', 'pos', 'variant_id', 'r2', 'lead_variant']]
            plink_ld_variants['chr'] = 'chr' + plink_ld_variants['chr'].astype(str)
            plink_ld_variants = plink_ld_variants.sort_values(by=['variant_id', 'r2'], ascending=False).drop_duplicates(subset='variant_id')
            print("plink_ld_variants:")
            print(plink_ld_variants.head())
            print(plink_ld_variants.shape)

            all_plink_variants = pd.concat([lead_variants, plink_ld_variants])
            all_plink_variants = all_plink_variants[['variant_id', 'r2', 'lead_variant']]
            all_plink_variants = all_plink_variants.sort_values( by=['variant_id', 'r2'], ascending=False)
            print("all_plink_variants:")
            print(all_plink_variants.head())
            print(all_plink_variants.shape)

            variant_scores = variant_scores.merge(all_plink_variants,
                on=['variant_id'],
                how='left')


    os.remove(tmp_bed_file_path)

    print()
    print(variant_scores.head())
    print("Annotation table shape:", variant_scores.shape)
    print()

    out_file = output_prefix + ".annotations.tsv"

    variant_scores.to_csv(out_file,\
                          sep="\t",\
                          index=False)
    print("Output written to:", out_file)

    print("DONE")


if __name__ == "__main__":
    main()
