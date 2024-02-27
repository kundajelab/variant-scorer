import pandas as pd
import numpy as np
import os
import subprocess

from utils.argmanager import *
from utils.helpers import *


def main(args = None):
    if args is None:
        args = fetch_variant_annotation_args()
    print(args)
    variant_scores_file = args.list
    output_prefix = args.out_prefix
    peak_path = args.peaks
    genes = args.genes

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


    if args.genes:

        print("annotating with closest genes")
        closest_gene_path = "%s.closest_genes.bed"%output_prefix
        gene_bedtools_intersect_cmd = "bedtools closest -d -t first -k 3 -a %s -b %s > %s"%(tmp_bed_file_path, genes, closest_gene_path)
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
        closest_gene_df['closest_gene_1'] = closest_gene_df['variant_id'].apply(lambda x: closest_genes[x][0] if len(closest_genes[x]) > 0 else '.')
        closest_gene_df['gene_distance_1'] = closest_gene_df['variant_id'].apply(lambda x: gene_dists[x][0] if len(closest_genes[x]) > 0 else '.')

        closest_gene_df['closest_gene_2'] = closest_gene_df['variant_id'].apply(lambda x: closest_genes[x][1] if len(closest_genes[x]) > 1 else '.')
        closest_gene_df['gene_distance_2'] = closest_gene_df['variant_id'].apply(lambda x: gene_dists[x][1] if len(closest_genes[x]) > 1 else '.')

        closest_gene_df['closest_gene_3'] = closest_gene_df['variant_id'].apply(lambda x: closest_genes[x][2] if len(closest_genes[x]) > 2 else '.')
        closest_gene_df['gene_distance_3'] = closest_gene_df['variant_id'].apply(lambda x: gene_dists[x][2] if len(closest_genes[x]) > 2 else '.')

        closest_gene_df = closest_gene_df[['variant_id', 'closest_gene_1', 'gene_distance_1',
                                           'closest_gene_2', 'gene_distance_2',
                                           'closest_gene_3', 'gene_distance_3']]
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

    os.remove(tmp_bed_file_path)

    print()
    print(variant_scores.head())
    print("Annotation table shape:", variant_scores.shape)
    print()

    out_file = output_prefix + ".annotations.tsv"
    variant_scores.to_csv(out_file,\
                          sep="\t",\
                          index=False)

    print("DONE")


if __name__ == "__main__":
    main()
