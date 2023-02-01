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
import statsmodels.stats.multitest
import math

SNP_SCHEMA = {'original': ["chr", "pos", "rsid", "allele1", "allele2"],
              'plink': ["chr", "rsid", "ignore1", "pos", "allele1", "allele2"],
              'bed': ['chr', 'start', 'pos', 'allele1', 'allele2', 'rsid', 'snp_id'],
              'chrombpnet': ["chr", "pos", "allele1", "allele2", "rsid"]}

PEAK_SCHEMA = {'narrowpeak': ['chr', 'start', 'end', 3, 4, 5, 6, 7, 'rank', 'summit']}


def main():
    args = argmanager.fetch_scoring_args()
    print(args)

    np.random.seed(args.random_seed)

    out_dir = os.path.sep.join(args.out_prefix.split(os.path.sep)[:-1])
    if not os.path.exists(out_dir):
        raise OSError("Output directory does not exist")

    # load the model
    model = load_model_wrapper(args.model)

    # load the variants
    variants_table = pd.read_csv(args.list, header=None, sep='\t', names=SNP_SCHEMA[args.schema])
    variants_table.drop(columns=[str(x) for x in variants_table.columns if str(x).startswith('ignore')], inplace=True)
    variants_table['chr'] = variants_table['chr'].astype(str).str.lower()
    has_chr_prefix = any('chr' in x for x in variants_table['chr'].tolist())
    if not has_chr_prefix:
        variants_table['chr'] = 'chr' + variants_table['chr']

    chrom_sizes = pd.read_csv(args.chrom_sizes, header=None, sep='\t', names=['chrom', 'size'])
    chrom_sizes_dict = chrom_sizes.set_index('chrom')['size'].to_dict()

    print("Original variants table shape:", variants_table.shape)

    if args.chrom:
        variants_table = variants_table.loc[variants_table['chr'] == args.chrom]
        print("Chromosome variants table shape:", variants_table.shape)

    # infer input length
    if args.lite:
        input_len = model.input_shape[0][1]
    else:
        input_len = model.input_shape[1]

    print("Input length inferred from the model:", input_len)

    variants_table = variants_table.loc[variants_table.apply(lambda x: get_valid_variants(x.chr, x.pos, x.allele1, x.allele2, input_len, chrom_sizes_dict), axis=1)]
    variants_table.reset_index(drop=True, inplace=True)

    print("Final variants table shape:", variants_table.shape)

    if args.total_shuf:
        if len(variants_table) > args.total_shuf:
            shuf_variants_table = variants_table.sample(args.total_shuf,
                                                        random_state=args.random_seed,
                                                        ignore_index=True,
                                                        replace=False)
        else:
            shuf_variants_table = variants_table.sample(args.total_shuf,
                                                        random_state=args.random_seed,
                                                        ignore_index=True,
                                                        replace=True)
    else:
        total_shuf = len(variants_table) * args.num_shuf
        shuf_variants_table = variants_table.sample(args.total_shuf,
                                                    random_state=args.random_seed,
                                                    ignore_index=True,
                                                    replace=True)

    shuf_variants_table['random_seed'] = np.random.permutation(len(shuf_variants_table))

    print("Shuffled variants table shape:", shuf_variants_table.shape)

    if len(shuf_variants_table) > 0:
        if args.debug_mode:
            shuf_variants_table = shuf_variants_table.sample(10000, random_state=args.random_seed, ignore_index=True)
            print()
            print(shuf_variants_table.head())
            print("Debug shuffled variants table shape:", shuf_variants_table.shape)
            print()

        shuf_rsids, shuf_allele1_pred_counts, shuf_allele2_pred_counts, \
        shuf_allele1_pred_profiles, shuf_allele2_pred_profiles = fetch_variant_predictions(model,
                                                                            shuf_variants_table,
                                                                            input_len,
                                                                            args.genome,
                                                                            args.batch_size,
                                                                            debug_mode=args.debug_mode,
                                                                            lite=args.lite,
                                                                            shuf=True,
                                                                            num_shuf=args.num_shuf)

    if args.peaks:
        if args.peak_chrom_sizes == None:
            args.peak_chrom_sizes = args.chrom_sizes
        if args.peak_genome == None:
            args.peak_genome = args.genome

        peak_chrom_sizes = pd.read_csv(args.peak_chrom_sizes, header=None, sep='\t', names=['chrom', 'size'])
        peak_chrom_sizes_dict = peak_chrom_sizes.set_index('chrom')['size'].to_dict()

        peaks = pd.read_csv(args.peaks, header=None, sep='\t', names=PEAK_SCHEMA['narrowpeak'])

        print("Original peak table shape:", peaks.shape)

        peaks.sort_values(by=['chr', 'start', 'end', 'summit', 'rank'], ascending=[True, True, True, True, False], inplace=True)
        peaks.drop_duplicates(subset=['chr', 'start', 'end', 'summit'], inplace=True)
        peaks = peaks.loc[peaks.apply(lambda x: get_valid_peaks(x.chr, x.start, x.summit, input_len, peak_chrom_sizes_dict), axis=1)]
        peaks.reset_index(drop=True, inplace=True)

        print("De-duplicated peak table shape:", peaks.shape)

        if args.debug_mode:
            peaks = peaks.sample(10000, random_state=args.random_seed, ignore_index=True)
            print()
            print(peaks.head())
            print("Debug peak table shape:", peaks.shape)
            print()

        if args.max_peaks:
            if len(peaks) > args.max_peaks:
                peaks = peaks.sample(args.max_peaks, random_state=args.random_seed, ignore_index=True)
                print("Subsampled peak table shape:", peaks.shape)

        pred_counts, pred_profiles = fetch_peak_predictions(model,
                                                            peaks,
                                                            input_len,
                                                            args.peak_genome,
                                                            args.batch_size,
                                                            debug_mode=args.debug_mode,
                                                            lite=args.lite)

        if len(shuf_variants_table) > 0:
            shuf_logfc, shuf_jsd, \
            shuf_allele1_percentile, shuf_allele2_percentile = get_variant_scores_with_peaks(shuf_allele1_pred_counts,
                                                                                             shuf_allele2_pred_counts,
                                                                                             shuf_allele1_pred_profiles,
                                                                                             shuf_allele2_pred_profiles,
                                                                                             pred_counts)

            shuf_max_percentile = np.maximum(shuf_allele1_percentile, shuf_allele2_percentile)
            shuf_percentile_change = shuf_allele2_percentile - shuf_allele1_percentile
            shuf_abs_logfc = np.squeeze(np.abs(shuf_logfc))
            shuf_abs_logfc_jsd = shuf_abs_logfc * shuf_jsd
            shuf_abs_logfc_jsd_max_percentile = shuf_abs_logfc_jsd * shuf_max_percentile

    else:
        if len(shuf_variants_table) > 0:
            shuf_logfc, shuf_jsd = get_variant_scores(shuf_allele1_pred_counts,
                                                      shuf_allele2_pred_counts,
                                                      shuf_allele1_pred_profiles,
                                                      shuf_allele2_pred_profiles)
            shuf_abs_logfc = np.squeeze(np.abs(shuf_logfc))
            shuf_abs_logfc_jsd = shuf_abs_logfc * shuf_jsd

    todo_chroms = [x for x in variants_table.chr.unique() if not os.path.exists('.'.join([args.out_prefix, x, "variant_predictions.h5"]))]

    for chrom in todo_chroms:
        print()
        print(chrom)
        print()

        chrom_variants_table = variants_table.loc[variants_table['chr'] == chrom].sort_values(by='pos').copy()
        chrom_variants_table.reset_index(drop=True, inplace=True)

        print(str(chrom) + " variants table shape:", chrom_variants_table.shape)
        print()

        if args.debug_mode:
            chrom_variants_table = chrom_variants_table.sample(10000, random_state=args.random_seed, ignore_index=True)
            print()
            print(chrom_variants_table.head())
            print("Debug " + str(chrom) + " variants table shape:", chrom_variants_table.shape)
            print()

        # fetch model prediction for variants
        rsids, allele1_pred_counts, allele2_pred_counts, \
        allele1_pred_profiles, allele2_pred_profiles = fetch_variant_predictions(model,
                                                                            chrom_variants_table,
                                                                            input_len,
                                                                            args.genome,
                                                                            args.batch_size,
                                                                            debug_mode=args.debug_mode,
                                                                            lite=args.lite,
                                                                            shuf=False,
                                                                            num_shuf=args.num_shuf)

        if args.peaks:
            logfc, jsd, \
            allele1_percentile, allele2_percentile = get_variant_scores_with_peaks(allele1_pred_counts,
                                                                                   allele2_pred_counts,
                                                                                   allele1_pred_profiles,
                                                                                   allele2_pred_profiles,
                                                                                   pred_counts)

        else:
            logfc, jsd = get_variant_scores(allele1_pred_counts,
                                                              allele2_pred_counts,
                                                              allele1_pred_profiles,
                                                              allele2_pred_profiles)

        # unpack rsids to write outputs and write score to output
        assert np.array_equal(chrom_variants_table["rsid"].tolist(), rsids)
        chrom_variants_table["allele1_pred_counts"] = allele1_pred_counts
        chrom_variants_table["allele2_pred_counts"] = allele2_pred_counts
        chrom_variants_table["logfc"] = logfc
        chrom_variants_table["abs_logfc"] = abs(chrom_variants_table["logfc"])
        chrom_variants_table["jsd"] = jsd
        chrom_variants_table["abs_logfc_x_jsd"] = chrom_variants_table["abs_logfc"] * chrom_variants_table["jsd"]

        if len(shuf_variants_table) > 0:
            chrom_variants_table["logfc_pval"] = chrom_variants_table["logfc"].apply(lambda x:
                                                                                     2 * min(scipy.stats.percentileofscore(shuf_logfc, x) / 100,
                                                                                             1 - (scipy.stats.percentileofscore(shuf_logfc, x) / 100)))
            chrom_variants_table["jsd_pval"] = chrom_variants_table["jsd"].apply(lambda x:
                                                                             1 - (scipy.stats.percentileofscore(shuf_jsd, x) / 100))
            chrom_variants_table["abs_logfc_x_jsd_pval"] = chrom_variants_table["abs_logfc_x_jsd"].apply(lambda x:
                                                                             1 - (scipy.stats.percentileofscore(shuf_abs_logfc_jsd, x) / 100))

        if args.peaks:
            chrom_variants_table["allele1_percentile"] = allele1_percentile
            chrom_variants_table["allele2_percentile"] = allele2_percentile
            chrom_variants_table["max_percentile"] = chrom_variants_table[["allele1_percentile", "allele2_percentile"]].max(axis=1)
            chrom_variants_table["percentile_change"] = chrom_variants_table["allele2_percentile"] - chrom_variants_table["allele1_percentile"]
            chrom_variants_table["abs_logfc_x_jsd_x_max_percentile"] = chrom_variants_table["abs_logfc_x_jsd"] * chrom_variants_table["max_percentile"]

            if len(shuf_variants_table) > 0:
                chrom_variants_table["max_percentile_pval"] = chrom_variants_table["max_percentile"].apply(lambda x:
                                                                             1 - (scipy.stats.percentileofscore(shuf_max_percentile, x) / 100))
                chrom_variants_table["percentile_change_pval"] = chrom_variants_table["percentile_change"].apply(lambda x:
                                                                                         2 * min(scipy.stats.percentileofscore(shuf_percentile_change, x) / 100,
                                                                                                 1 - (scipy.stats.percentileofscore(shuf_percentile_change, x) / 100)))
                chrom_variants_table["abs_logfc_x_jsd_x_max_percentile_pval"] = chrom_variants_table["abs_logfc_x_jsd_x_max_percentile"].apply(lambda x:
                                                            1 - (scipy.stats.percentileofscore(shuf_abs_logfc_jsd_max_percentile, x) / 100))

        print()
        print(chrom_variants_table.head())
        print("Output " + str(chrom) + " score table shape:", chrom_variants_table.shape)
        print()

        chrom_variants_table.to_csv('.'.join([args.out_prefix, chrom, "variant_scores.tsv"]), sep="\t", index=False)

        # store predictions at variants
        if not args.no_hdf5:
            with h5py.File('.'.join([args.out_prefix, chrom, "variant_predictions.h5"]), 'w') as f:
                observed = f.create_group('observed')
                observed.create_dataset('allele1_pred_counts', data=allele1_pred_counts, compression='gzip', compression_opts=9)
                observed.create_dataset('allele2_pred_counts', data=allele2_pred_counts, compression='gzip', compression_opts=9)
                observed.create_dataset('allele1_pred_profiles', data=allele1_pred_profiles, compression='gzip', compression_opts=9)
                observed.create_dataset('allele2_pred_profiles', data=allele2_pred_profiles, compression='gzip', compression_opts=9)
                if len(shuf_variants_table) > 0:
                    shuffled = f.create_group('shuffled')
                    shuffled.create_dataset('shuf_allele1_pred_counts', data=shuf_allele1_pred_counts, compression='gzip', compression_opts=9)
                    shuffled.create_dataset('shuf_allele2_pred_counts', data=shuf_allele2_pred_counts, compression='gzip', compression_opts=9)
                    shuffled.create_dataset('shuf_logfc', data=shuf_logfc, compression='gzip', compression_opts=9)
                    shuffled.create_dataset('shuf_abs_logfc', data=shuf_abs_logfc, compression='gzip', compression_opts=9)
                    shuffled.create_dataset('shuf_jsd', data=shuf_jsd, compression='gzip', compression_opts=9)
                    shuffled.create_dataset('shuf_abs_logfc_x_jsd', data=shuf_abs_logfc_jsd, compression='gzip', compression_opts=9)
                    if args.peaks:
                        shuffled.create_dataset('shuf_max_percentile', data=shuf_max_percentile, compression='gzip', compression_opts=9)
                        shuffled.create_dataset('shuf_percentile_change', data=shuf_percentile_change, compression='gzip', compression_opts=9)
                        shuffled.create_dataset('shuf_abs_logfc_x_jsd_x_max_percentile', data=shuf_abs_logfc_jsd_max_percentile, compression='gzip', compression_opts=9)

        print("DONE:", str(chrom))
        print()


def get_valid_peaks(chrom, pos, summit, input_len, chrom_sizes_dict):
    flank = input_len // 2
    lower_check = ((pos + summit) - flank > 0)
    upper_check = ((pos + summit) + flank <= chrom_sizes_dict[chrom])
    in_bounds = lower_check and upper_check
    return in_bounds

def get_valid_variants(chrom, pos, allele1, allele2, input_len, chrom_sizes_dict):
    flank = input_len // 2
    lower_check = (pos - flank > 0)
    upper_check = (pos + flank <= chrom_sizes_dict[chrom])
    in_bounds = lower_check and upper_check
    no_allele1_indel = (len(allele1) == 1)
    no_allele2_indel = (len(allele2) == 1)
    no_indels = no_allele1_indel and no_allele2_indel
    valid_variants = in_bounds and no_indels
    return valid_variants

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

def fetch_peak_predictions(model, peaks, input_len, genome_fasta, batch_size, debug_mode=False, lite=False):
    pred_counts = []
    pred_profiles = []

    # peak sequence generator
    peak_gen = PeakGenerator(peaks=peaks,
                             input_len=input_len,
                             genome_fasta=genome_fasta,
                             batch_size=batch_size,
                             debug_mode=debug_mode)

    for i in tqdm(range(len(peak_gen))):

        seqs = peak_gen[i]

        if lite:
            batch_preds = model.predict([seqs,
                                         np.zeros((len(seqs), model.output_shape[0][1])),
                                         np.zeros((len(seqs), ))],
                                        verbose=False)
        else:
            batch_preds = model.predict(seqs, verbose=False)

        batch_preds[1] = np.array([batch_preds[1][i] for i in range(len(batch_preds[1]))])
        pred_counts.extend(np.exp(batch_preds[1]))
        pred_profiles.extend(np.squeeze(softmax(batch_preds[0])))


    pred_counts = np.array(pred_counts)
    pred_profiles = np.array(pred_profiles)

    return pred_counts, pred_profiles

def fetch_variant_predictions(model, variants_table, input_len, genome_fasta, batch_size, debug_mode=False, lite=False, shuf=False, num_shuf=10):
    rsids = []
    allele1_pred_counts = []
    allele2_pred_counts = []
    allele1_pred_profiles = []
    allele2_pred_profiles = []

    # snp sequence generator
    snp_gen = SNPGenerator(variants_table=variants_table,
                           input_len=input_len,
                           genome_fasta=genome_fasta,
                           batch_size=batch_size,
                           debug_mode=debug_mode,
                           shuf=shuf,
                           num_shuf=num_shuf)

    for i in tqdm(range(len(snp_gen))):

        batch_rsids, allele1_seqs, allele2_seqs = snp_gen[i]

        if lite:
            allele1_batch_preds = model.predict([allele1_seqs,
                                                 np.zeros((len(allele1_seqs), model.output_shape[0][1])),
                                                 np.zeros((len(allele1_seqs), ))],
                                                verbose=False)
            allele2_batch_preds = model.predict([allele2_seqs,
                                                 np.zeros((len(allele2_seqs), model.output_shape[0][1])),
                                                 np.zeros((len(allele2_seqs), ))],
                                                verbose=False)
        else:
            allele1_batch_preds = model.predict(allele1_seqs, verbose=False)
            allele2_batch_preds = model.predict(allele2_seqs, verbose=False)

        allele1_batch_preds[1] = np.array([allele1_batch_preds[1][i] for i in range(len(allele1_batch_preds[1]))])
        allele2_batch_preds[1] = np.array([allele2_batch_preds[1][i] for i in range(len(allele2_batch_preds[1]))])

        allele1_pred_counts.extend(np.exp(allele1_batch_preds[1]))
        allele2_pred_counts.extend(np.exp(allele2_batch_preds[1]))

        allele1_pred_profiles.extend(np.squeeze(softmax(allele1_batch_preds[0])))
        allele2_pred_profiles.extend(np.squeeze(softmax(allele2_batch_preds[0])))

        rsids.extend(batch_rsids)

    rsids = np.array(rsids)
    allele1_pred_counts = np.array(allele1_pred_counts)
    allele2_pred_counts = np.array(allele2_pred_counts)
    allele1_pred_profiles = np.array(allele1_pred_profiles)
    allele2_pred_profiles = np.array(allele2_pred_profiles)

    return rsids, allele1_pred_counts, allele2_pred_counts, \
           allele1_pred_profiles, allele2_pred_profiles

def get_variant_scores_with_peaks(allele1_pred_counts, allele2_pred_counts,
                       allele1_pred_profiles, allele2_pred_profiles, pred_counts):
    logfc = np.log2(allele2_pred_counts / allele1_pred_counts)
    jsd = np.array([jensenshannon(x,y) for x,y in zip(allele2_pred_profiles, allele1_pred_profiles)])
    allele1_percentile = np.array([np.mean(pred_counts < x) for x in allele1_pred_counts])
    allele2_percentile = np.array([np.mean(pred_counts < x) for x in allele2_pred_counts])

    return logfc, jsd, allele1_percentile, allele2_percentile

def get_variant_scores(allele1_pred_counts, allele2_pred_counts,
                       allele1_pred_profiles, allele2_pred_profiles):
    logfc = np.log2(allele2_pred_counts / allele1_pred_counts)
    jsd = np.array([jensenshannon(x,y) for x,y in zip(allele2_pred_profiles, allele1_pred_profiles)])

    return logfc, jsd

if __name__ == "__main__":
    main()
