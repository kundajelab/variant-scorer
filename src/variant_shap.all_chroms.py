from turtle import pos
from snp_generator import SNPGenerator
from peak_generator import PeakGenerator
from utils import argmanager, losses
import scipy.stats
from scipy.spatial.distance import jensenshannon
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import load_model
import tensorflow as tf
import pandas as pd
import os
import argparse
import numpy as np
import h5py
import psutil
from tqdm import tqdm
import shap
from shap_utils import *
tf.compat.v1.disable_v2_behavior()


SCHEMA = {'bed': ["chr", "pos", "rsid", "allele1", "allele2"],
          'plink': ["chr", "rsid", "ignore1", "pos", "allele1", "allele2"],
          'narrowpeak': ['chr', 'start', 'end', 3, 4, 5, 6, 7, 'rank', 'summit']}

def main():
    args = argmanager.fetch_scoring_args()
    print(args)

    out_dir = os.path.sep.join(args.out_prefix.split(os.path.sep)[:-1])
    print()
    print('out_dir:', out_dir)
    print()
    if not os.path.exists(out_dir):
        raise OSError("Output directory does not exist")

    # load the model
    model = load_model_wrapper(args.model)

    # load the variants
    variants_table = pd.read_csv(args.list, header=None, sep='\t', names=SCHEMA[args.schema])
    variants_table.drop(columns=[x for x in variants_table.columns if x.startswith('ignore')], inplace=True)
    variants_table['chr'] = variants_table['chr'].astype(str)
    has_chr_prefix = any('chr' in x for x in variants_table['chr'].tolist())
    if not has_chr_prefix:
        variants_table['chr'] = 'chr' + variants_table['chr']

    chrom_sizes = pd.read_csv(args.chrom_sizes, header=None, sep='\t', names=['chrom', 'size'])
    chrom_sizes_dict = chrom_sizes.set_index('chrom')['size'].to_dict()

    if args.debug_mode:
        variants_table = variants_table.sample(10)
        print(variants_table.head())

    # infer input length
    if args.lite:
        input_len = model.input_shape[0][1]
    else:
        input_len = model.input_shape[1]
    print("input length inferred from the model: ", input_len)

    print(variants_table.shape)
    variants_table = variants_table.loc[variants_table.apply(lambda x: get_valid_variants(x.chr, x.pos, input_len, chrom_sizes_dict), axis=1)]
    variants_table.reset_index(drop=True, inplace=True)
    print(variants_table.shape)

    # fetch model prediction for variants
    rsids, allele1_counts_shap, allele2_counts_shap = fetch_shap(model,
                                                                 variants_table,
                                                                 input_len,
                                                                 args.genome,
                                                                 args.batch_size,
                                                                 debug_mode=args.debug_mode,
                                                                 lite=args.lite,
                                                                 bias=None,
                                                                 shuf=False)

    # store shap at variants
    with h5py.File('.'.join([args.out_prefix, "variant_shap.h5"]), 'w') as f:
        wo_bias = f.create_group('wo_bias')
        wo_bias.create_dataset('allele1_counts_shap', data=allele1_counts_shap)
        wo_bias.create_dataset('allele2_counts_shap', data=allele2_counts_shap)

    print("DONE")

def get_valid_variants(chrom, pos, input_len, chrom_sizes_dict):
    flank = input_len // 2
    lower_check = (pos - flank > 0)
    upper_check = (pos + flank <= chrom_sizes_dict[chrom])
    in_bounds = lower_check and upper_check
    return in_bounds

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

def fetch_shap(model, variants_table, input_len, genome_fasta, batch_size, debug_mode=False, lite=False, bias=None, shuf=False):
    rsids = []
    allele1_counts_shap = []
    allele2_counts_shap = []

    # snp sequence generator
    snp_gen = SNPGenerator(variants_table=variants_table,
                           input_len=input_len,
                           genome_fasta=genome_fasta,
                           batch_size=batch_size,
                           debug_mode=debug_mode,
                           shuf=shuf)

    for i in tqdm(range(len(snp_gen))):

        batch_rsids, allele1_seqs, allele2_seqs = snp_gen[i]

        if lite:
            counts_model_input = [model.input[0], model.input[2]]
            allele1_input = [allele1_seqs, np.zeros((allele1_seqs.shape[0], 1))]
            allele2_input = [allele2_seqs, np.zeros((allele2_seqs.shape[0], 1))]

            profile_model_counts_explainer = shap.explainers.deep.TFDeepExplainer(
                (counts_model_input, tf.reduce_sum(model.outputs[1], axis=-1)),
                shuffle_several_times,
                combine_mult_and_diffref=combine_mult_and_diffref)

            allele1_counts_shap_batch = profile_model_counts_explainer.shap_values(
                allele1_input, progress_message=10)
            allele2_counts_shap_batch = profile_model_counts_explainer.shap_values(
                allele2_input, progress_message=10)

            allele1_counts_shap_batch = allele1_counts_shap_batch[0] * allele1_input[0]
            allele2_counts_shap_batch = allele2_counts_shap_batch[0] * allele2_input[0]

        else:
            counts_model_input = model.input
            allele1_input = allele1_seqs
            allele2_input = allele2_seqs

            profile_model_counts_explainer = shap.explainers.deep.TFDeepExplainer(
                (counts_model_input, tf.reduce_sum(model.outputs[1], axis=-1)),
                shuffle_several_times,
                combine_mult_and_diffref=combine_mult_and_diffref)

            allele1_counts_shap_batch = profile_model_counts_explainer.shap_values(
                allele1_input, progress_message=10)
            allele2_counts_shap_batch = profile_model_counts_explainer.shap_values(
                allele2_input, progress_message=10)

            allele1_counts_shap_batch = allele1_counts_shap_batch * allele1_input
            allele2_counts_shap_batch = allele2_counts_shap_batch * allele2_input

        allele1_counts_shap.extend(allele1_counts_shap_batch)
        allele2_counts_shap.extend(allele2_counts_shap_batch)

        rsids.extend(batch_rsids)

    return np.array(rsids), np.array(allele1_counts_shap), np.array(allele2_counts_shap)


if __name__ == "__main__":
    main()




