import os
import numpy as np
import h5py


def merge_h5_predictions(chrom_h5_files, merged_h5_file):
    """Read per-chromosome prediction H5 files and concatenate into one merged file.

    Reads allele1_pred_counts, allele2_pred_counts, allele1_pred_profiles, and
    allele2_pred_profiles from each file's 'observed' group, concatenates them, and
    writes the result to merged_h5_file. Each per-chromosome file is deleted after reading.

    Args:
        chrom_h5_files: ordered list of paths to per-chromosome H5 files
        merged_h5_file: path for the output merged H5 file
    """
    allele1_pred_counts = []
    allele2_pred_counts = []
    allele1_pred_profiles = []
    allele2_pred_profiles = []

    for chrom_preds_file in chrom_h5_files:
        if os.path.isfile(chrom_preds_file):
            with h5py.File(chrom_preds_file, 'r') as chrom_h5:
                if 'observed' in chrom_h5:
                    observed = chrom_h5['observed']
                    if 'allele1_pred_counts' in observed:
                        allele1_pred_counts.append(observed['allele1_pred_counts'][:])
                    if 'allele2_pred_counts' in observed:
                        allele2_pred_counts.append(observed['allele2_pred_counts'][:])
                    if 'allele1_pred_profiles' in observed:
                        allele1_pred_profiles.append(observed['allele1_pred_profiles'][:])
                    if 'allele2_pred_profiles' in observed:
                        allele2_pred_profiles.append(observed['allele2_pred_profiles'][:])
                    print(f"Removing {chrom_preds_file}...")
                    os.remove(chrom_preds_file)
                else:
                    print(f"Warning: 'observed' group not found in {chrom_preds_file}, skipping")
        else:
            print(f"Warning: {chrom_preds_file} not found, skipping")

    if allele1_pred_counts and allele2_pred_counts and allele1_pred_profiles and allele2_pred_profiles:
        with h5py.File(merged_h5_file, 'w') as merged_h5:
            observed = merged_h5.create_group('observed')
            observed.create_dataset('allele1_pred_counts', data=np.concatenate(allele1_pred_counts), compression='gzip', compression_opts=9)
            observed.create_dataset('allele2_pred_counts', data=np.concatenate(allele2_pred_counts), compression='gzip', compression_opts=9)
            observed.create_dataset('allele1_pred_profiles', data=np.concatenate(allele1_pred_profiles), compression='gzip', compression_opts=9)
            observed.create_dataset('allele2_pred_profiles', data=np.concatenate(allele2_pred_profiles), compression='gzip', compression_opts=9)
