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
from generators.variant_generator import VariantGenerator
from generators.peak_generator import PeakGenerator
from utils import argmanager, losses
from utils.helpers import *

def main():
    args = argmanager.fetch_scoring_args()
    print(args)

    np.random.seed(args.random_seed)
    if args.forward_only:
        print("running variant scoring only for forward sequences")
    
    out_dir = os.path.sep.join(args.out_prefix.split(os.path.sep)[:-1])
    if not os.path.exists(out_dir):
        raise OSError("Output directory does not exist")

    # load the model and variants
    model = load_model_wrapper(args.model)
    variants_table=load_variant_table(args.list, args.schema)
    variants_table = variants_table.fillna('-')
    
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

    shuf_variants_table = create_shuffle_table(variants_table,args.random_seed, args.total_shuf, args.num_shuf)
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
                                                                            forward_only=args.forward_only)

    if args.peaks:
        if args.peak_chrom_sizes == None:
            args.peak_chrom_sizes = args.chrom_sizes
        if args.peak_genome == None:
            args.peak_genome = args.genome

        peak_chrom_sizes = pd.read_csv(args.peak_chrom_sizes, header=None, sep='\t', names=['chrom', 'size'])
        peak_chrom_sizes_dict = peak_chrom_sizes.set_index('chrom')['size'].to_dict()

        peaks = pd.read_csv(args.peaks, header=None, sep='\t', names=get_peak_schema('narrowpeak'))

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
                                                            lite=args.lite,
                                                            forward_only=args.forward_only)

        if len(shuf_variants_table) > 0:
            shuf_logfc, shuf_jsd, \
            shuf_allele1_percentile, shuf_allele2_percentile = get_variant_scores_with_peaks(shuf_allele1_pred_counts,
                                                                                            shuf_allele2_pred_counts,
                                                                                            shuf_allele1_pred_profiles,
                                                                                            shuf_allele2_pred_profiles,
                                                                                            pred_counts)
            _, shuf_jsd = adjust_indel_jsd(shuf_variants_table,
                                        shuf_allele1_pred_profiles,
                                        shuf_allele2_pred_profiles,
                                        shuf_jsd)
            
            shuf_max_percentile = np.maximum(shuf_allele1_percentile, shuf_allele2_percentile)
            shuf_percentile_change = shuf_allele2_percentile - shuf_allele1_percentile
            shuf_abs_percentile_change = np.abs(shuf_percentile_change)
            shuf_abs_logfc = np.abs(shuf_logfc)
            shuf_abs_logfc_jsd = shuf_abs_logfc * shuf_jsd
            shuf_abs_logfc_jsd_max_percentile = shuf_abs_logfc_jsd * shuf_max_percentile

            assert shuf_abs_logfc.shape == shuf_logfc.shape
            assert shuf_abs_logfc.shape == shuf_jsd.shape
            assert shuf_abs_logfc.shape == shuf_abs_logfc_jsd.shape
            assert shuf_abs_logfc_jsd_max_percentile.shape == shuf_abs_logfc_jsd.shape
            assert shuf_abs_logfc_jsd_max_percentile.shape == shuf_max_percentile.shape
            assert shuf_max_percentile.shape == shuf_allele1_percentile.shape
            assert shuf_max_percentile.shape == shuf_allele2_percentile.shape
            assert shuf_max_percentile.shape == shuf_percentile_change.shape
            assert shuf_abs_percentile_change.shape == shuf_percentile_change.shape

    else:
        if len(shuf_variants_table) > 0:
            shuf_logfc, shuf_jsd = get_variant_scores(shuf_allele1_pred_counts,
                                                    shuf_allele2_pred_counts,
                                                    shuf_allele1_pred_profiles,
                                                    shuf_allele2_pred_profiles)
            
            _, shuf_jsd = adjust_indel_jsd(shuf_variants_table,
                                shuf_allele1_pred_profiles,
                                shuf_allele2_pred_profiles,
                                shuf_jsd)
                
            shuf_abs_logfc = np.squeeze(np.abs(shuf_logfc))
            shuf_abs_logfc_jsd = shuf_abs_logfc * shuf_jsd

    if not args.no_hdf5:
        todo_chroms = [x for x in variants_table.chr.unique() if not os.path.exists('.'.join([args.out_prefix, str(x), "variant_predictions.h5"]))]
    else:
        todo_chroms = [x for x in variants_table.chr.unique() if not os.path.exists('.'.join([args.out_prefix, str(x), "variant_scores.tsv"]))]

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
            print("Debug variants table shape:", chrom_variants_table.shape)
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
                                                                            forward_only=args.forward_only)

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

        indel_idx, adjusted_jsd_list = adjust_indel_jsd(chrom_variants_table,allele1_pred_profiles,allele2_pred_profiles,jsd)
        have_indel_variants = (len(indel_idx) > 0)

        # unpack rsids to write outputs and write score to output
        assert np.array_equal(chrom_variants_table["rsid"].tolist(), rsids)
        chrom_variants_table["allele1_pred_counts"] = allele1_pred_counts
        chrom_variants_table["allele2_pred_counts"] = allele2_pred_counts
        chrom_variants_table["logfc"] = logfc
        chrom_variants_table["abs_logfc"] = abs(chrom_variants_table["logfc"])
        if have_indel_variants:
            chrom_variants_table["jsd"] = adjusted_jsd_list
        else:
            chrom_variants_table["jsd"] = jsd
        chrom_variants_table["original_jsd"] = jsd
        chrom_variants_table["abs_logfc_x_jsd"] = chrom_variants_table["abs_logfc"] * chrom_variants_table["jsd"]

        if len(shuf_variants_table) > 0:
            chrom_variants_table["abs_logfc_pval"] = get_pvals(chrom_variants_table["abs_logfc"].tolist(), shuf_abs_logfc)
            chrom_variants_table["jsd_pval"] = get_pvals(chrom_variants_table["jsd"].tolist(), shuf_jsd)
            chrom_variants_table["abs_logfc_x_jsd_pval"] = get_pvals(chrom_variants_table["abs_logfc_x_jsd"].tolist(), shuf_abs_logfc_jsd)
        if args.peaks:
            chrom_variants_table["allele1_percentile"] = allele1_percentile
            chrom_variants_table["allele2_percentile"] = allele2_percentile
            chrom_variants_table["max_percentile"] = chrom_variants_table[["allele1_percentile", "allele2_percentile"]].max(axis=1)
            chrom_variants_table["percentile_change"] = chrom_variants_table["allele2_percentile"] - chrom_variants_table["allele1_percentile"]
            chrom_variants_table["abs_percentile_change"] = abs(chrom_variants_table["percentile_change"])
            chrom_variants_table["abs_logfc_x_jsd_x_max_percentile"] = chrom_variants_table["abs_logfc_x_jsd"] * chrom_variants_table["max_percentile"]

            if len(shuf_variants_table) > 0:
                chrom_variants_table["max_percentile_pval"] = get_pvals(chrom_variants_table["max_percentile"].tolist(), shuf_max_percentile)
                chrom_variants_table["abs_percentile_change_pval"] = get_pvals(chrom_variants_table["abs_percentile_change"].tolist(), shuf_abs_percentile_change)
                chrom_variants_table["abs_logfc_x_jsd_x_max_percentile_pval"] = get_pvals(chrom_variants_table["abs_logfc_x_jsd_x_max_percentile"].tolist(), shuf_abs_logfc_jsd_max_percentile)

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
                        shuffled.create_dataset('shuf_abs_percentile_change', data=shuf_abs_percentile_change, compression='gzip', compression_opts=9)
                        shuffled.create_dataset('shuf_abs_logfc_x_jsd_x_max_percentile', data=shuf_abs_logfc_jsd_max_percentile, compression='gzip', compression_opts=9)

    print("DONE")
    print()


if __name__ == "__main__":
    main()
