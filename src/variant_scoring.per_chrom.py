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
    variants_table = load_variant_table(args.list, args.schema)
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

    shuf_scores_file = '.'.join([args.out_prefix, "variant_scores.shuffled.tsv"])
    peak_scores_file = '.'.join([args.out_prefix, "peak_scores.tsv"])

    if len(shuf_variants_table) > 0:
        if args.debug_mode:
            shuf_variants_table = shuf_variants_table.sample(10000, random_state=args.random_seed, ignore_index=True)
            print()
            print(shuf_variants_table.head())
            print("Debug shuffled variants table shape:", shuf_variants_table.shape)
            print()

        shuf_variants_done = False
        if os.path.isfile(shuf_scores_file):
            shuf_variants_table_loaded = pd.read_table(shuf_scores_file)
            if shuf_variants_table_loaded['variant_id'].tolist() == shuf_variants_table['variant_id'].tolist():
                shuf_variants_table = shuf_variants_table_loaded.copy()
                shuf_variants_done = True

        if not shuf_variants_done:
            shuf_variant_ids, shuf_allele1_pred_counts, shuf_allele2_pred_counts, \
            shuf_allele1_pred_profiles, shuf_allele2_pred_profiles = fetch_variant_predictions(model,
                                                                                shuf_variants_table,
                                                                                input_len,
                                                                                args.genome,
                                                                                args.batch_size,
                                                                                debug_mode=args.debug_mode,
                                                                                lite=args.lite,
                                                                                shuf=True,
                                                                                forward_only=args.forward_only)
            assert np.array_equal(shuf_variants_table["variant_id"].tolist(), shuf_variant_ids)
            shuf_variants_table["allele1_pred_counts"] = shuf_allele1_pred_counts
            shuf_variants_table["allele2_pred_counts"] = shuf_allele2_pred_counts

            print()
            print(shuf_variants_table.head())
            print("Shuffled score table shape:", shuf_variants_table.shape)
            print()
            shuf_variants_table.to_csv(shuf_scores_file, sep="\t", index=False)

    if args.peaks:
        if args.peak_chrom_sizes == None:
            args.peak_chrom_sizes = args.chrom_sizes
        if args.peak_genome == None:
            args.peak_genome = args.genome

        peak_chrom_sizes = pd.read_csv(args.peak_chrom_sizes, header=None, sep='\t', names=['chrom', 'size'])
        peak_chrom_sizes_dict = peak_chrom_sizes.set_index('chrom')['size'].to_dict()

        peaks = pd.read_csv(args.peaks, header=None, sep='\t', names=get_peak_schema('narrowpeak'))
        peaks['peak_id'] = peaks['chr'] + ':' + peaks['start'].astype(str) + '-' + peaks['end'].astype(str)

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

        peak_scores_done = False
        if os.path.isfile(peak_scores_file):
            peaks_loaded = pd.read_table(peak_scores_file)
            if peaks_loaded['peak_id'].tolist() == peaks['peak_id'].tolist():
                peaks = peaks_loaded.copy()
                peak_scores_done = True
            
        if not peak_scores_done:
            peak_ids, peak_pred_counts, peak_pred_profiles = fetch_peak_predictions(model,
                                                                peaks,
                                                                input_len,
                                                                args.peak_genome,
                                                                args.batch_size,
                                                                debug_mode=args.debug_mode,
                                                                lite=args.lite,
                                                                forward_only=args.forward_only)
            assert np.array_equal(peaks["peak_id"].tolist(), peak_ids)
            peaks["peak_score"] = peak_pred_counts
            print()
            print(peaks.head())
            print("Peak score table shape:", peaks.shape)
            print()
            peaks.to_csv(peak_scores_file, sep="\t", index=False)

        if len(shuf_variants_table) > 0:
            shuf_logfc, shuf_jsd, \
            shuf_allele1_percentile, shuf_allele2_percentile = get_variant_scores_with_peaks(shuf_allele1_pred_counts,
                                                                                            shuf_allele2_pred_counts,
                                                                                            shuf_allele1_pred_profiles,
                                                                                            shuf_allele2_pred_profiles,
                                                                                            peak_pred_counts)
            shuf_indel_idx, shuf_adjusted_jsd_list = adjust_indel_jsd(shuf_variants_table,
                                                                      shuf_allele1_pred_profiles,
                                                                      shuf_allele2_pred_profiles,
                                                                      shuf_jsd)
            shuf_has_indel_variants = (len(shuf_indel_idx) > 0)
            
            shuf_variants_table["logfc"] = shuf_logfc
            shuf_variants_table["abs_logfc"] = np.abs(shuf_logfc)
            if shuf_has_indel_variants:
                shuf_variants_table["jsd"] = shuf_adjusted_jsd_list
            else:
                shuf_variants_table["jsd"] = shuf_jsd
                assert shuf_adjusted_jsd_list == shuf_jsd
            shuf_variants_table['original_jsd'] = shuf_jsd
            shuf_variants_table["logfc_x_jsd"] =  shuf_variants_table["logfc"] * shuf_variants_table["jsd"]
            shuf_variants_table["abs_logfc_x_jsd"] = shuf_variants_table["abs_logfc"] * shuf_variants_table["jsd"]

            shuf_variants_table["allele1_percentile"] = shuf_allele1_percentile
            shuf_variants_table["allele2_percentile"] = shuf_allele2_percentile
            shuf_variants_table["max_percentile"] = shuf_variants_table[["allele1_percentile", "allele2_percentile"]].max(axis=1)
            shuf_variants_table["percentile_change"] = shuf_variants_table["allele2_percentile"] - shuf_variants_table["allele1_percentile"]
            shuf_variants_table["abs_percentile_change"] = np.abs(shuf_variants_table["percentile_change"])
            shuf_variants_table["logfc_x_max_percentile"] = shuf_variants_table["logfc"] * shuf_variants_table["max_percentile"]
            shuf_variants_table["abs_logfc_x_max_percentile"] = shuf_variants_table["abs_logfc"] * shuf_variants_table["max_percentile"]
            shuf_variants_table["jsd_x_max_percentile"] = shuf_variants_table["jsd"] * shuf_variants_table["max_percentile"]
            shuf_variants_table["logfc_x_jsd_x_max_percentile"] = shuf_variants_table["logfc_x_jsd"] * shuf_variants_table["max_percentile"]
            shuf_variants_table["abs_logfc_x_jsd_x_max_percentile"] = shuf_variants_table["abs_logfc_x_jsd"] * shuf_variants_table["max_percentile"]

            assert shuf_variants_table["abs_logfc"].shape == shuf_logfc.shape
            assert shuf_variants_table["abs_logfc"].shape == shuf_jsd.shape
            assert shuf_variants_table["abs_logfc"].shape == shuf_variants_table["abs_logfc_x_jsd"].shape

            print()
            print(shuf_variants_table.head())
            print("Shuffled score table shape:", shuf_variants_table.shape)
            print()
            shuf_variants_table.to_csv(shuf_scores_file, sep="\t", index=False)

    else:
        if len(shuf_variants_table) > 0:
            shuf_logfc, shuf_jsd = get_variant_scores(shuf_allele1_pred_counts,
                                                    shuf_allele2_pred_counts,
                                                    shuf_allele1_pred_profiles,
                                                    shuf_allele2_pred_profiles)
            
            shuf_indel_idx, shuf_adjusted_jsd_list = adjust_indel_jsd(shuf_variants_table,
                                                                      shuf_allele1_pred_profiles,
                                                                      shuf_allele2_pred_profiles,
                                                                      shuf_jsd)
            shuf_has_indel_variants = (len(shuf_indel_idx) > 0)
            
            shuf_variants_table["logfc"] = shuf_logfc
            shuf_variants_table["abs_logfc"] = np.abs(shuf_logfc)
            if shuf_has_indel_variants:
                shuf_variants_table["jsd"] = shuf_adjusted_jsd_list
            else:
                shuf_variants_table["jsd"] = shuf_jsd
                assert shuf_adjusted_jsd_list == shuf_jsd
            shuf_variants_table['original_jsd'] = shuf_jsd
            shuf_variants_table["logfc_x_jsd"] =  shuf_variants_table["logfc"] * shuf_variants_table["jsd"]
            shuf_variants_table["abs_logfc_x_jsd"] = shuf_variants_table["abs_logfc"] * shuf_variants_table["jsd"]

            assert shuf_variants_table["abs_logfc"].shape == shuf_logfc.shape
            assert shuf_variants_table["abs_logfc"].shape == shuf_jsd.shape
            assert shuf_variants_table["abs_logfc"].shape == shuf_variants_table["abs_logfc_x_jsd"].shape

            print()
            print(shuf_variants_table.head())
            print("Shuffled score table shape:", shuf_variants_table.shape)
            print()
            shuf_variants_table.to_csv(shuf_scores_file, sep="\t", index=False)

    todo_chroms = [x for x in variants_table.chr.unique() if not os.path.exists('.'.join([args.out_prefix, str(x), "variant_scores.tsv"]))]

    for chrom in todo_chroms:
        print()
        print(chrom)
        print()

        chrom_variants_table = variants_table.loc[variants_table['chr'] == chrom].sort_values(by='pos').copy()
        chrom_variants_table.reset_index(drop=True, inplace=True)

        chrom_scores_done = False
        chrom_scores_file = '.'.join([args.out_prefix, str(chrom), "variant_scores.tsv"])
        if os.path.isfile(chrom_scores_file):
            chrom_variants_table_loaded = pd.read_table(chrom_scores_file)
            if chrom_variants_table_loaded['variant_id'].tolist() == chrom_variants_table['variant_id'].tolist():
                chrom_scores_done = True

        if not chrom_scores_done:
            print(str(chrom) + " variants table shape:", chrom_variants_table.shape)
            print()

            if args.debug_mode:
                chrom_variants_table = chrom_variants_table.sample(10000, random_state=args.random_seed, ignore_index=True)
                print()
                print(chrom_variants_table.head())
                print("Debug variants table shape:", chrom_variants_table.shape)
                print()

            # fetch model prediction for variants
            variant_ids, allele1_pred_counts, allele2_pred_counts, \
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
                                                                                        peak_pred_counts)

            else:
                logfc, jsd = get_variant_scores(allele1_pred_counts,
                                                allele2_pred_counts,
                                                allele1_pred_profiles,
                                                allele2_pred_profiles)

            indel_idx, adjusted_jsd_list = adjust_indel_jsd(chrom_variants_table,allele1_pred_profiles,allele2_pred_profiles,jsd)
            has_indel_variants = (len(indel_idx) > 0)

            assert np.array_equal(chrom_variants_table["variant_id"].tolist(), variant_ids)
            chrom_variants_table["allele1_pred_counts"] = allele1_pred_counts
            chrom_variants_table["allele2_pred_counts"] = allele2_pred_counts
            chrom_variants_table["logfc"] = logfc
            chrom_variants_table["abs_logfc"] = abs(chrom_variants_table["logfc"])
            if has_indel_variants:
                chrom_variants_table["jsd"] = adjusted_jsd_list
            else:
                chrom_variants_table["jsd"] = jsd
            chrom_variants_table["original_jsd"] = jsd
            chrom_variants_table["logfc_x_jsd"] = chrom_variants_table["logfc"] * chrom_variants_table["jsd"]
            chrom_variants_table["abs_logfc_x_jsd"] = chrom_variants_table["abs_logfc"] * chrom_variants_table["jsd"]

            if len(shuf_variants_table) > 0:
                chrom_variants_table["logfc.pval"] = get_pvals(chrom_variants_table["logfc"].tolist(), shuf_variants_table["logfc"], tail="both")
                chrom_variants_table["abs_logfc.pval"] = get_pvals(chrom_variants_table["abs_logfc"].tolist(), shuf_variants_table["abs_logfc"], tail="right")
                chrom_variants_table["jsd.pval"] = get_pvals(chrom_variants_table["jsd"].tolist(), shuf_variants_table["jsd"], tail="right")
                chrom_variants_table["logfc_x_jsd.pval"] = get_pvals(chrom_variants_table["logfc_x_jsd"].tolist(), shuf_variants_table["logfc_x_jsd"], tail="both")
                chrom_variants_table["abs_logfc_x_jsd.pval"] = get_pvals(chrom_variants_table["abs_logfc_x_jsd"].tolist(), shuf_variants_table["abs_logfc_x_jsd"], tail="right")
            if args.peaks:
                chrom_variants_table["allele1_percentile"] = allele1_percentile
                chrom_variants_table["allele2_percentile"] = allele2_percentile
                chrom_variants_table["max_percentile"] = chrom_variants_table[["allele1_percentile", "allele2_percentile"]].max(axis=1)
                chrom_variants_table["percentile_change"] = chrom_variants_table["allele2_percentile"] - chrom_variants_table["allele1_percentile"]
                chrom_variants_table["abs_percentile_change"] = abs(chrom_variants_table["percentile_change"])
                chrom_variants_table["logfc_x_max_percentile"] = chrom_variants_table["logfc"] * chrom_variants_table["max_percentile"]
                chrom_variants_table["abs_logfc_x_max_percentile"] = chrom_variants_table["abs_logfc"] * chrom_variants_table["max_percentile"]
                chrom_variants_table["jsd_x_max_percentile"] = chrom_variants_table["jsd"] * chrom_variants_table["max_percentile"]
                chrom_variants_table["logfc_x_jsd_x_max_percentile"] = chrom_variants_table["logfc_x_jsd"] * chrom_variants_table["max_percentile"]
                chrom_variants_table["abs_logfc_x_jsd_x_max_percentile"] = chrom_variants_table["abs_logfc_x_jsd"] * chrom_variants_table["max_percentile"]

                if len(shuf_variants_table) > 0:
                    chrom_variants_table["max_percentile.pval"] = get_pvals(chrom_variants_table["max_percentile"].tolist(),
                                                                            shuf_variants_table["max_percentile"], tail="right")
                    chrom_variants_table['percentile_change.pval'] = get_pvals(chrom_variants_table["percentile_change"].tolist(),
                                                                               shuf_variants_table["percentile_change"], tail="both")
                    chrom_variants_table["abs_percentile_change.pval"] = get_pvals(chrom_variants_table["abs_percentile_change"].tolist(),
                                                                                   shuf_variants_table["abs_percentile_change"], tail="right")
                    chrom_variants_table["logfc_x_max_percentile.pval"] = get_pvals(chrom_variants_table["logfc_x_max_percentile"].tolist(),
                                                                                    shuf_variants_table["logfc_x_max_percentile"], tail="both")
                    chrom_variants_table["abs_logfc_x_max_percentile.pval"] = get_pvals(chrom_variants_table["abs_logfc_x_max_percentile"].tolist(),
                                                                                        shuf_variants_table["abs_logfc_x_max_percentile"], tail="right")
                    chrom_variants_table["jsd_x_max_percentile.pval"] = get_pvals(chrom_variants_table["jsd_x_max_percentile"].tolist(),
                                                                                  shuf_variants_table["jsd_x_max_percentile"], tail="right")
                    chrom_variants_table["logfc_x_jsd_x_max_percentile.pval"] = get_pvals(chrom_variants_table["logfc_x_jsd_x_max_percentile"].tolist(),
                                                                                          shuf_variants_table["logfc_x_jsd_x_max_percentile"], tail="both")
                    chrom_variants_table["abs_logfc_x_jsd_x_max_percentile.pval"] = get_pvals(chrom_variants_table["abs_logfc_x_jsd_x_max_percentile"].tolist(),
                                                                                              shuf_variants_table["abs_logfc_x_jsd_x_max_percentile"], tail="right")

            if args.schema == "bed":
                chrom_variants_table['pos'] = chrom_variants_table['pos'] - 1
            
            print()
            print(chrom_variants_table.head())
            print("Output " + str(chrom) + " score table shape:", chrom_variants_table.shape)
            print()

            # store predictions at variants
            if not args.no_hdf5:
                with h5py.File('.'.join([args.out_prefix, chrom, "variant_predictions.h5"]), 'w') as f:
                    observed = f.create_group('observed')
                    observed.create_dataset('allele1_pred_counts', data=allele1_pred_counts, compression='gzip', compression_opts=9)
                    observed.create_dataset('allele2_pred_counts', data=allele2_pred_counts, compression='gzip', compression_opts=9)
                    observed.create_dataset('allele1_pred_profiles', data=allele1_pred_profiles, compression='gzip', compression_opts=9)
                    observed.create_dataset('allele2_pred_profiles', data=allele2_pred_profiles, compression='gzip', compression_opts=9)

            chrom_variants_table.to_csv(chrom_scores_file, sep="\t", index=False)

    print("DONE")
    print()


if __name__ == "__main__":
    main()
