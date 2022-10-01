from turtle import pos
from snp_generator import SNPGenerator
from utils import argmanager, losses
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

SCHEMA = {'bed': ["chr", "pos", "rsid", "allele1", "allele2"],
          'plink': ["chr", "rsid", "ignore1", "pos", "allele1", "allele2"],
          'narrowpeak': ['chr', 'start', 'end', 3, 4, 5, 6, 7, 'rank', 'summit'],
          'chrombpnet': ["chr", "pos", "allele1", "allele2", "rsid"]}

def main():
    args = argmanager.fetch_scoring_args()
    print(args)

    out_dir = os.path.sep.join(args.out_prefix.split(os.path.sep)[:-1])
    if not os.path.exists(out_dir):
        raise OSError("Output directory does not exist")

    # load the model
    model = load_model_wrapper(args.model)
    
    # load the variants
    variants_table = pd.read_csv(args.list, header=None, sep='\t', names=SCHEMA[args.schema])
    if args.schema == "chrombpnet":
        variants_table["pos"] += 1 
    variants_table.drop(columns=[x for x in variants_table.columns if x.startswith('ignore')], inplace=True)
    variants_table['chr'] = variants_table['chr'].astype(str)
    has_chr_prefix = any('chr' in x for x in variants_table['chr'].tolist())
    if not has_chr_prefix:
        variants_table['chr'] = 'chr' + variants_table['chr']

    chrom_sizes = pd.read_csv(args.chrom_sizes, header=None, sep='\t', names=['chrom', 'size'])
    chrom_sizes_dict = chrom_sizes.set_index('chrom')['size'].to_dict()

    if args.debug_mode:
        variants_table = variants_table.sample(100000)
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

    print()
    print("Filtered Variants")
    print()
    print(psutil.virtual_memory())
    print()

    todo_chroms = [x for x in variants_table.chr.unique() if not os.path.exists('.'.join([args.out_prefix, x, "variant_predictions.h5"]))]

    #assert len(todo_chroms) < 10

    # split by chromosome to fit predictions in memory
    for chrom in todo_chroms:
        print()
        print(chrom)
        print()
        print(psutil.virtual_memory())
        print()

        chrom_variants_table = variants_table.loc[variants_table['chr'] == chrom].sort_values(by='pos').copy()
        chrom_variants_table.reset_index(drop=True, inplace=True)
        print(chrom_variants_table.shape)
        print()

        # fetch model prediction for variants
        rsids, allele1_count_preds, allele2_count_preds, \
        allele1_profile_preds, allele2_profile_preds = fetch_variant_predictions(model,
                                                                        chrom_variants_table,
                                                                        input_len,
                                                                        args.genome,
                                                                        args.batch_size,
                                                                        debug_mode=args.debug_mode,
                                                                        lite=args.lite,
                                                                        bias=None)

        print()
        print("Got Predictions")
        print()
        print(psutil.virtual_memory())
        print()

        # get varaint effect scores
        log_fold_change, profile_jsd = get_variant_scores(allele1_count_preds, allele2_count_preds,
                                                        allele1_profile_preds, allele2_profile_preds)

        print()
        print("Got Scores")
        print()
        print(psutil.virtual_memory())
        print()

        # unpack rsids to write outputs and write score to output
        assert np.array_equal(chrom_variants_table["rsid"].tolist(), rsids)
        chrom_variants_table["log_fold_change"] = log_fold_change
        chrom_variants_table["profile_jsd"] = profile_jsd
        chrom_variants_table["allele1_pred_counts"] = allele1_count_preds
        chrom_variants_table["allele2_pred_counts"] = allele2_count_preds

        print()
        print("Added Scores to Table")
        print()
        print(psutil.virtual_memory())
        print()

        if args.bias != None:
            bias = load_model_wrapper(args.bias)
            w_bias_rsids, w_bias_allele1_count_preds, w_bias_allele2_count_preds, \
            w_bias_allele1_profile_preds, w_bias_allele2_profile_preds = fetch_variant_predictions(model,
                                                                                                chrom_variants_table,
                                                                                                input_len,
                                                                                                args.genome,
                                                                                                args.batch_size,
                                                                                                debug_mode=args.debug_mode,
                                                                                                lite=args.lite,
                                                                                                bias=bias)
            assert np.array_equal(chrom_variants_table["rsid"].tolist(), w_bias_rsids)
            chrom_variants_table["allele1_w_bias_pred_counts"] = w_bias_allele1_count_preds
            chrom_variants_table["allele2_w_bias_pred_counts"] = w_bias_allele2_count_preds

        print()
        print("Got Bias Predictions")
        print()
        print(psutil.virtual_memory())
        print()

        chrom_variants_table.to_csv('.'.join([args.out_prefix, chrom, "variant_scores.tsv"]), sep="\t", index=False)

        print()
        print("Wrote Score Table")
        print()
        print(psutil.virtual_memory())
        print()

        # store predictions at variants
        with h5py.File('.'.join([args.out_prefix, chrom, "variant_predictions.h5"]), 'w') as f:
            wo_bias = f.create_group('wo_bias')
            wo_bias.create_dataset('allele1_pred_counts', data=allele1_count_preds)
            wo_bias.create_dataset('allele2_pred_counts', data=allele2_count_preds)
            wo_bias.create_dataset('allele1_pred_profile', data=allele1_profile_preds)
            wo_bias.create_dataset('allele2_pred_profile', data=allele2_profile_preds)

            if args.bias != None:
                w_bias = f.create_group('w_bias')
                w_bias.create_dataset('allele1_w_bias_pred_counts', data=w_bias_allele1_count_preds)
                w_bias.create_dataset('allele2_w_bias_pred_counts', data=w_bias_allele2_count_preds)
                w_bias.create_dataset('allele1_w_bias_pred_profile', data=w_bias_allele1_profile_preds)
                w_bias.create_dataset('allele2_w_bias_pred_profile', data=w_bias_allele2_profile_preds)

        print()
        print("Wrote Predictions HDF5")
        print()
        print(psutil.virtual_memory())
        print()

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

def fetch_variant_predictions(model, variants_table, input_len, genome_fasta, batch_size, debug_mode=False, lite=False, bias=None):
    rsids = []
    allele1_count_preds = []
    allele2_count_preds = []
    allele1_profile_preds = []
    allele2_profile_preds = []

    # snp sequence generator 
    snp_gen = SNPGenerator(variants_table=variants_table,
                           input_len=input_len,
                           genome_fasta=genome_fasta,
                           batch_size=batch_size,
                           debug_mode=debug_mode)

    for i in tqdm(range(len(snp_gen))):

        batch_rsids, allele1_seqs, allele2_seqs = snp_gen[i]

        if lite:
            if bias != None:
                allele1_bias_batch_preds = bias.predict(allele1_seqs, verbose=False)
                allele2_bias_batch_preds = bias.predict(allele2_seqs, verbose=False)

                allele1_batch_preds = model.predict([allele1_seqs,
                                                     allele1_bias_batch_preds[0],
                                                     allele1_bias_batch_preds[1]],
                                                    verbose=False)
                allele2_batch_preds = model.predict([allele2_seqs,
                                                     allele2_bias_batch_preds[0],
                                                     allele2_bias_batch_preds[1]],
                                                    verbose=False)
            else:
                allele1_batch_preds = model.predict([allele1_seqs,
                                                     np.zeros((len(allele1_seqs), model.output_shape[0][1])),
                                                     np.zeros((len(allele1_seqs), ))],
                                                    verbose=False)
                allele2_batch_preds = model.predict([allele2_seqs,
                                                     np.zeros((len(allele2_seqs), model.output_shape[0][1])),
                                                     np.zeros((len(allele2_seqs), ))],
                                                    verbose=False)
        else:
            if bias != None:
                allele1_batch_preds = bias.predict(allele1_seqs, verbose=False)
                allele2_batch_preds = bias.predict(allele2_seqs, verbose=False)
            else:
                allele1_batch_preds = model.predict(allele1_seqs, verbose=False)
                allele2_batch_preds = model.predict(allele2_seqs, verbose=False)

        allele1_count_preds.extend(np.exp(np.squeeze(allele1_batch_preds[1])) - 1)
        allele2_count_preds.extend(np.exp(np.squeeze(allele2_batch_preds[1])) - 1)

        allele1_profile_preds.extend(np.squeeze(softmax(allele1_batch_preds[0])))
        allele2_profile_preds.extend(np.squeeze(softmax(allele2_batch_preds[0])))

        rsids.extend(batch_rsids)

    return np.array(rsids), np.array(allele1_count_preds), np.array(allele2_count_preds), \
           np.array(allele1_profile_preds), np.array(allele2_profile_preds)

def get_variant_scores(allele1_count_preds, allele2_count_preds, allele1_profile_preds, allele2_profile_preds):
    log_fold_change = np.log2(allele2_count_preds / allele1_count_preds)
    profile_jsd_diff = np.array([jensenshannon(x,y) for x,y in zip(allele2_profile_preds, allele1_profile_preds)])

    return log_fold_change, profile_jsd_diff

if __name__ == "__main__":
    main()




