from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import load_model
import tensorflow as tf
import scipy.stats
from scipy.spatial.distance import jensenshannon
import pandas as pd
import os
import argparse
import numpy as np
import h5py
import math
from tqdm import tqdm
import sys
sys.path.append('..')
from generators.variant_generator import VariantGenerator
from utils import argmanager, losses

def load_variant_table(table_path, schema):
    variants_table = pd.read_csv(table_path, header=None, sep='\t', names=get_variant_schema(schema))
    variants_table.drop(columns=[str(x) for x in variants_table.columns if str(x).startswith('ignore')], inplace=True)
    variants_table['chr'] = variants_table['chr'].astype(str)
    has_chr_prefix = any('chr' in x.lower() for x in variants_table['chr'].tolist())
    if not has_chr_prefix:
        variants_table['chr'] = 'chr' + variants_table['chr']
    if schema == "bed":
        variants_table['pos'] = variants_table['pos'] + 1
    return variants_table

def create_shuffle_table(variants_table,random_seed=None,total_shuf=None, num_shuf=None):
    if total_shuf:
        if len(variants_table) > total_shuf:
            shuf_variants_table = variants_table.sample(total_shuf,
                                                        random_state=random_seed,
                                                        ignore_index=True,
                                                        replace=False)
        else:
            shuf_variants_table = variants_table.sample(total_shuf,
                                                        random_state=random_seed,
                                                        ignore_index=True,
                                                        replace=True) 
        shuf_variants_table['random_seed'] = np.random.permutation(len(shuf_variants_table))
    else:
        if num_shuf:
            total_shuf = len(variants_table) * num_shuf
            shuf_variants_table = variants_table.sample(total_shuf,
                                                        random_state=random_seed,
                                                        ignore_index=True,
                                                        replace=True)
            shuf_variants_table['random_seed'] = np.random.permutation(len(shuf_variants_table))
        else:
            ## empty dataframe
            shuf_variants_table = pd.DataFrame()
    return shuf_variants_table

def get_variant_schema(schema):
    var_SCHEMA = {'original': ["chr", "pos", "rsid", "allele1", "allele2"],
                  'plink': ["chr", "rsid", "ignore1", "pos", "allele1", "allele2"],
                  'bed': ['chr', 'pos', 'end', 'allele1', 'allele2', 'rsid'],
                  'chrombpnet': ["chr", "pos", "allele1", "allele2", "rsid"]}
    return var_SCHEMA[schema]

def get_valid_variants(chrom, pos, allele1, allele2, input_len, chrom_sizes_dict):
    valid_chrom = chrom in chrom_sizes_dict
    if valid_chrom:
        flank = input_len // 2
        lower_check = (pos - flank > 0)
        upper_check = (pos + flank <= chrom_sizes_dict[chrom])
        in_bounds = lower_check and upper_check
        # no_allele1_indel = (len(allele1) == 1)
        # no_allele2_indel = (len(allele2) == 1)
        # no_indel = no_allele1_indel and no_allele2_indel
        # valid_variant = valid_chrom and in_bounds and no_indel
        valid_variant = valid_chrom and in_bounds
        return valid_variant
    else:
        return False

def softmax(x, temp=1):
    norm_x = x - np.mean(x,axis=1, keepdims=True)
    return np.exp(temp*norm_x)/np.sum(np.exp(temp*norm_x), axis=1, keepdims=True)

def load_model_wrapper(model_file):
    # read .h5 model
    custom_objects = {"multinomial_nll": losses.multinomial_nll, "tf": tf}
    get_custom_objects().update(custom_objects)
    model = load_model(model_file, compile=False)
    print("model loaded succesfully")
    return model

def fetch_variant_predictions(model, variants_table, input_len, genome_fasta, batch_size, debug_mode=False, shuf=False, forward_only=False,upstream_flank=None,downstream_flank=None):
    rsids = []
    allele1_pred_counts = []
    allele2_pred_counts = []
    if not forward_only:
        revcomp_allele1_pred_counts = []
        revcomp_allele2_pred_counts = []
    # variant sequence generator
    var_gen = VariantGenerator(variants_table=variants_table,
                           input_len=input_len,
                           genome_fasta=genome_fasta,
                           batch_size=batch_size,
                           debug_mode=False,
                           shuf=shuf,
                           upstream_flank=upstream_flank,
                           downstream_flank=downstream_flank)

    for i in tqdm(range(len(var_gen))):

        batch_rsids, allele1_seqs, allele2_seqs = var_gen[i]
        revcomp_allele1_seqs = allele1_seqs[:, ::-1, ::-1]
        revcomp_allele2_seqs = allele2_seqs[:, ::-1, ::-1]
    
        allele1_batch_preds = model.predict(allele1_seqs, verbose=False)
        allele2_batch_preds = model.predict(allele2_seqs, verbose=False)
        if not forward_only:
            revcomp_allele1_batch_preds = model.predict(revcomp_allele1_seqs, verbose=False)
            revcomp_allele2_batch_preds = model.predict(revcomp_allele2_seqs, verbose=False)

        allele1_pred_counts.extend(allele1_batch_preds)
        allele2_pred_counts.extend(allele2_batch_preds)
        
        if not forward_only:
            revcomp_allele1_pred_counts.extend(allele1_batch_preds)
            revcomp_allele2_pred_counts.extend(allele2_batch_preds)

        rsids.extend(batch_rsids)

    rsids = np.array(rsids)
    allele1_pred_counts = np.array(allele1_pred_counts)
    allele2_pred_counts = np.array(allele2_pred_counts)
    if not forward_only:
        revcomp_allele1_pred_counts = np.array(revcomp_allele1_pred_counts)
        revcomp_allele2_pred_counts = np.array(revcomp_allele2_pred_counts)
        average_allele1_pred_counts = np.average([allele1_pred_counts,revcomp_allele1_pred_counts],axis=0)
        average_allele2_pred_counts = np.average([allele2_pred_counts,revcomp_allele2_pred_counts],axis=0)
        return rsids, average_allele1_pred_counts, average_allele2_pred_counts
    else:
        return rsids, allele1_pred_counts, allele2_pred_counts

def get_variant_scores(allele1_pred_counts, allele2_pred_counts):
    variant_score = allele2_pred_counts - allele1_pred_counts   

    return variant_score

def get_pvals(obs, bg):
    sorted_obs = np.sort(obs)[::-1]
    sorted_obs_indices = np.argsort(obs)[::-1]
    sorted_obs_indices = np.argsort(sorted_obs_indices)
    sorted_obs_indices_list = sorted_obs_indices.astype(int).tolist()
    sorted_bg = np.sort(bg)[::-1]

    bg_pointer = 0
    bg_len = len(sorted_bg)
    sorted_pvals = []

    for val in sorted_obs:
        while val <= sorted_bg[bg_pointer] and bg_pointer != bg_len - 1:
            bg_pointer += 1
        sorted_pvals.append((bg_pointer + 1) / (bg_len + 1))

    sorted_pvals = np.array(sorted_pvals)
    pvals = sorted_pvals[sorted_obs_indices_list]

    return pvals
