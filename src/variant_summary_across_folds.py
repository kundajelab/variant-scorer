import pandas as pd
import numpy as np
import os
import deepdish
from multiprocessing import Pool
from functools import partial
import copy

from sklearn.metrics import average_precision_score
import scipy.stats
from scipy.stats import mannwhitneyu
from scipy.stats import hypergeom
from scipy.stats import fisher_exact
from scipy.stats import wilcoxon

from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import os
from utils.argmanager import *
from utils.helpers import *

def geo_mean_overflow(iterable,axis=0):
    return np.exp(np.log(iterable).mean(axis=0))

def main():
    args = fetch_variant_summary_args()
    print(args)
    variant_score_path = args.score_path
    peak_path = args.peaks
    output_prefix = args.out_prefix
    genes = args.genes
    variant_table_list = args.score_list

    score_dict = {}
    for i in range(len(variant_table_list)):
        variant_score_file = os.path.join(variant_score_path, variant_table_list[i])
        assert os.path.isfile(variant_score_file)
        var_score = pd.read_table(variant_score_file)
        score_dict[i] = var_score

    variant_scores = score_dict[0][get_variant_schema(args.schema)].copy()

    for score in ["logfc"]:
        if score in score_dict[0]:
            variant_scores.loc[:, (score + '.mean')] = np.mean(np.array([score_dict[fold][score].tolist()
                                                                    for fold in score_dict]), axis=0)
    for score in ["abs_logfc", "jsd", "abs_logfc_x_jsd", "abs_logfc_x_max_percentile", "jsd_x_max_percentile", "abs_logfc_x_jsd_x_max_percentile"]:
        if score in score_dict[0]:
            variant_scores.loc[:, (score + '.mean')] = np.mean(np.array([score_dict[fold][score].tolist()
                                                                    for fold in score_dict]), axis=0)

            variant_scores.loc[:, (score + '.mean' + '.pval')] = geo_mean_overflow([score_dict[fold][score + '_pval'].values for fold in score_dict])

    tmp_bed_file_path = output_prefix + ".variant_table.tmp.bed"

    if args.schema == "bed":
        if variant_scores['pos'].equals(variant_scores['end']):
            variant_scores['pos'] = variant_scores['pos'] - 1
        variant_scores_bed_format = variant_scores[['chr','pos','end','allele1','allele2','rsid']].copy()
    else:
        ### convert to bed format
        variant_scores_bed_format = variant_scores[['chr','pos','allele1','allele2','rsid']].copy()
        variant_scores_bed_format['pos']  = variant_scores_bed_format.apply(lambda x: int(x.pos)-1, axis = 1)
        variant_scores_bed_format['end']  = variant_scores_bed_format.apply(lambda x: int(x.pos)+len(x.allele1), axis = 1)
        variant_scores_bed_format = variant_scores_bed_format[['chr','pos','end','allele1','allele2','rsid']]
        variant_scores_bed_format = variant_scores_bed_format.sort_values(["chr","pos","end"])

    variant_scores_bed_format.to_csv(tmp_bed_file_path,\
                                sep="\t",\
                                header=None,\
                                index=False)

    import subprocess
    print("annotating with closest genes")
    closest_gene_path="%s.closest_genes.bed"%output_prefix
    gene_bedtools_intersect_cmd = "bedtools closest -d -t first -k 3 -a %s -b %s > %s"%(tmp_bed_file_path, genes, closest_gene_path)
    _ = subprocess.call(gene_bedtools_intersect_cmd,\
                shell=True)

    closest_gene_df = pd.read_csv(closest_gene_path, sep='\t', header=None)

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

    closest_gene_df.drop_duplicates(subset=5, inplace=True)
    closest_gene_df = closest_gene_df.rename({5:'rsid'},axis=1)
    closest_gene_df = closest_gene_df[['rsid']]
    closest_gene_df['closest_gene_1'] = closest_gene_df['rsid'].apply(lambda x: closest_genes[x][0] if len(closest_genes[x]) > 0 else '.')
    closest_gene_df['gene_distance_1'] = closest_gene_df['rsid'].apply(lambda x: gene_dists[x][0] if len(closest_genes[x]) > 0 else '.')

    closest_gene_df['closest_gene_2'] = closest_gene_df['rsid'].apply(lambda x: closest_genes[x][1] if len(closest_genes[x]) > 1 else '.')
    closest_gene_df['gene_distance_2'] = closest_gene_df['rsid'].apply(lambda x: gene_dists[x][1] if len(closest_genes[x]) > 1 else '.')

    closest_gene_df['closest_gene_3'] = closest_gene_df['rsid'].apply(lambda x: closest_genes[x][2] if len(closest_genes[x]) > 2 else '.')
    closest_gene_df['gene_distance_3'] = closest_gene_df['rsid'].apply(lambda x: gene_dists[x][2] if len(closest_genes[x]) > 2 else '.')

    print("annotating with peak overlap")
    peak_intersect_path="%s.peak_overlap.bed"%output_prefix
    peak_bedtools_intersect_cmd="bedtools intersect -wa -u -a %s -b %s > %s"%(tmp_bed_file_path, peak_path,peak_intersect_path)
    _ = subprocess.call(peak_bedtools_intersect_cmd,\
                shell=True)
    peak_intersect_df=pd.read_table(peak_intersect_path, sep='\t', header=None)
    os.remove(tmp_bed_file_path)

    variant_scores['peak_overlap'] = False
    column_idx = variant_scores.columns.get_loc("peak_overlap")

    variant_scores.iloc[np.where(variant_scores['rsid'].isin(peak_intersect_df[5]))[0],column_idx] = True
    variant_scores = variant_scores.merge(closest_gene_df,on='rsid', how='inner')

    print()
    print(variant_scores.head())
    print("Summary score table shape:", variant_scores.shape)
    print()

    out_file = output_prefix + ".mean.variant_scores.peak_overlap.closest_genes.tsv"
    variant_scores.to_csv(out_file,\
                          sep="\t",\
                          index=False)

    print("DONE")


if __name__ == "__main__":
    main()
