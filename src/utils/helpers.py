from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import load_model
import tensorflow as tf
from scipy.spatial.distance import jensenshannon
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('..')
from generators.variant_generator import VariantGenerator
from generators.peak_generator import PeakGenerator
from utils import losses
from scipy.stats import binomtest
from math import ceil


def get_variant_schema(schema):
    var_SCHEMA = {'original': ['chr', 'pos', 'variant_id', 'allele1', 'allele2'],
                  'plink': ['chr', 'variant_id', 'ignore1', 'pos', 'allele1', 'allele2'],
                  'plink2': ['chr', 'variant_id', 'pos', 'allele1', 'allele2'],
                  'bed': ['chr', 'pos', 'end', 'allele1', 'allele2', 'variant_id'],
                  'chrombpnet': ['chr', 'pos', 'allele1', 'allele2', 'variant_id']}
    return var_SCHEMA[schema]

def get_peak_schema(schema):
    PEAK_SCHEMA = {'narrowpeak': ['chr', 'start', 'end', 'peak_id', 'peak_score',
                                  5, 6, 7, 'rank', 'summit']}
    return PEAK_SCHEMA[schema]

def get_valid_peaks(chrom, pos, summit, input_len, chrom_sizes_dict):
    valid_chrom = chrom in chrom_sizes_dict
    if valid_chrom:
        flank = input_len // 2
        lower_check = ((pos + summit) - flank > 0)
        upper_check = ((pos + summit) + flank <= chrom_sizes_dict[chrom])
        in_bounds = lower_check and upper_check
        valid_peak = valid_chrom and in_bounds
        return valid_peak
    else:
        return False

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
    norm_x = x - np.mean(x, axis=1, keepdims=True)
    return np.exp(temp*norm_x)/np.sum(np.exp(temp*norm_x), axis=1, keepdims=True)

def load_model_wrapper(model_file):
    # read .h5 model
    custom_objects = {"multinomial_nll": losses.multinomial_nll, "tf": tf}
    get_custom_objects().update(custom_objects)
    model = load_model(model_file, compile=False)
    print("model loaded succesfully")
    return model

def fetch_peak_predictions(model, peaks, input_len, genome_fasta, batch_size, debug_mode=False, lite=False,forward_only=False):
    peak_ids = []
    pred_counts = []
    pred_profiles = []
    if not forward_only:
        revcomp_counts = []
        revcomp_profiles = []

    # peak sequence generator
    peak_gen = PeakGenerator(peaks=peaks,
                             input_len=input_len,
                             genome_fasta=genome_fasta,
                             batch_size=batch_size,
                             debug_mode=debug_mode)

    for i in tqdm(range(len(peak_gen))):
        batch_peak_ids, seqs = peak_gen[i]
        revcomp_seq = seqs[:, ::-1, ::-1]

        if lite:
            batch_preds = model.predict([seqs,
                                         np.zeros((len(seqs), model.output_shape[0][1])),
                                         np.zeros((len(seqs), ))],
                                        verbose=False)

            if not forward_only:
                revcomp_batch_preds = model.predict([revcomp_seq,
                                             np.zeros((len(revcomp_seq), model.output_shape[0][1])),
                                             np.zeros((len(revcomp_seq), ))],
                                            verbose=False)
        else:
            batch_preds = model.predict(seqs, verbose=False)
            if not forward_only:
                revcomp_batch_preds = model.predict(revcomp_seq, verbose=False)

        batch_preds[1] = np.array([batch_preds[1][i] for i in range(len(batch_preds[1]))])
        pred_counts.extend(np.exp(batch_preds[1]))
        pred_profiles.extend(np.array(batch_preds[0]))   # np.squeeze(softmax()) to get probability profile

        if not forward_only:
            revcomp_batch_preds[1] = np.array([revcomp_batch_preds[1][i] for i in range(len(revcomp_batch_preds[1]))])
            revcomp_counts.extend(np.exp(revcomp_batch_preds[1]))
            revcomp_profiles.extend(np.array(revcomp_batch_preds[0]))    # np.squeeze(softmax()) to get probability profile

        peak_ids.extend(batch_peak_ids)

    peak_ids = np.array(peak_ids)
    pred_counts = np.array(pred_counts)
    pred_profiles = np.array(pred_profiles)

    if not forward_only:
        revcomp_counts = np.array(revcomp_counts)
        revcomp_profiles = np.array(revcomp_profiles)
        average_counts = np.average([pred_counts,revcomp_counts],axis=0)
        average_profiles = np.average([pred_profiles,revcomp_profiles[:,::-1]],axis=0)
        return peak_ids,average_counts,average_profiles
    else:
        return peak_ids,pred_counts,pred_profiles

def fetch_variant_predictions(model, variants_table, input_len, genome_fasta, batch_size, debug_mode=False, lite=False, shuf=False, forward_only=False):
    variant_ids = []
    allele1_pred_counts = []
    allele2_pred_counts = []
    allele1_pred_profiles = []
    allele2_pred_profiles = []
    if not forward_only:
        revcomp_allele1_pred_counts = []
        revcomp_allele2_pred_counts = []
        revcomp_allele1_pred_profiles = []
        revcomp_allele2_pred_profiles = []

    # variant sequence generator
    var_gen = VariantGenerator(variants_table=variants_table,
                           input_len=input_len,
                           genome_fasta=genome_fasta,
                           batch_size=batch_size,
                           debug_mode=False,
                           shuf=shuf)

    for i in tqdm(range(len(var_gen))):

        batch_variant_ids, allele1_seqs, allele2_seqs = var_gen[i]
        revcomp_allele1_seqs = allele1_seqs[:, ::-1, ::-1]
        revcomp_allele2_seqs = allele2_seqs[:, ::-1, ::-1]

        if lite:
            allele1_batch_preds = model.predict([allele1_seqs,
                                                 np.zeros((len(allele1_seqs), model.output_shape[0][1])),
                                                 np.zeros((len(allele1_seqs), ))],
                                                verbose=False)
            allele2_batch_preds = model.predict([allele2_seqs,
                                                 np.zeros((len(allele2_seqs), model.output_shape[0][1])),
                                                 np.zeros((len(allele2_seqs), ))],
                                                verbose=False)

            if not forward_only:
                revcomp_allele1_batch_preds = model.predict([revcomp_allele1_seqs,
                                                     np.zeros((len(revcomp_allele1_seqs), model.output_shape[0][1])),
                                                     np.zeros((len(revcomp_allele1_seqs), ))],
                                                    verbose=False)
                revcomp_allele2_batch_preds = model.predict([revcomp_allele2_seqs,
                                         np.zeros((len(revcomp_allele2_seqs), model.output_shape[0][1])),
                                         np.zeros((len(revcomp_allele2_seqs), ))],
                                        verbose=False)
        else:
            allele1_batch_preds = model.predict(allele1_seqs, verbose=False)
            allele2_batch_preds = model.predict(allele2_seqs, verbose=False)
            if not forward_only:
                revcomp_allele1_batch_preds = model.predict(revcomp_allele1_seqs, verbose=False)
                revcomp_allele2_batch_preds = model.predict(revcomp_allele2_seqs, verbose=False)

        allele1_batch_preds[1] = np.array([allele1_batch_preds[1][i] for i in range(len(allele1_batch_preds[1]))])
        allele2_batch_preds[1] = np.array([allele2_batch_preds[1][i] for i in range(len(allele2_batch_preds[1]))])
        allele1_pred_counts.extend(np.exp(allele1_batch_preds[1]))
        allele2_pred_counts.extend(np.exp(allele2_batch_preds[1]))
        allele1_pred_profiles.extend(np.array(allele1_batch_preds[0]))   # np.squeeze(softmax()) to get probability profile
        allele2_pred_profiles.extend(np.array(allele2_batch_preds[0]))

        if not forward_only:
            revcomp_allele1_batch_preds[1] = np.array([revcomp_allele1_batch_preds[1][i] for i in range(len(revcomp_allele1_batch_preds[1]))])
            revcomp_allele2_batch_preds[1] = np.array([revcomp_allele2_batch_preds[1][i] for i in range(len(revcomp_allele2_batch_preds[1]))])
            revcomp_allele1_pred_counts.extend(np.exp(revcomp_allele1_batch_preds[1]))
            revcomp_allele2_pred_counts.extend(np.exp(revcomp_allele2_batch_preds[1]))
            revcomp_allele1_pred_profiles.extend(np.array(revcomp_allele1_batch_preds[0]))   # np.squeeze(softmax()) to get probability profile
            revcomp_allele2_pred_profiles.extend(np.array(revcomp_allele2_batch_preds[0]))

        variant_ids.extend(batch_variant_ids)

    variant_ids = np.array(variant_ids)
    allele1_pred_counts = np.array(allele1_pred_counts)
    allele2_pred_counts = np.array(allele2_pred_counts)
    allele1_pred_profiles = np.array(allele1_pred_profiles)
    allele2_pred_profiles = np.array(allele2_pred_profiles)

    if not forward_only:
        revcomp_allele1_pred_counts = np.array(revcomp_allele1_pred_counts)
        revcomp_allele2_pred_counts = np.array(revcomp_allele2_pred_counts)
        revcomp_allele1_pred_profiles = np.array(revcomp_allele1_pred_profiles)
        revcomp_allele2_pred_profiles = np.array(revcomp_allele2_pred_profiles)
        average_allele1_pred_counts = np.average([allele1_pred_counts,revcomp_allele1_pred_counts],axis=0)
        average_allele1_pred_profiles = np.average([allele1_pred_profiles,revcomp_allele1_pred_profiles[:,::-1]],axis=0)
        average_allele2_pred_counts = np.average([allele2_pred_counts,revcomp_allele2_pred_counts],axis=0)
        average_allele2_pred_profiles = np.average([allele2_pred_profiles,revcomp_allele2_pred_profiles[:,::-1]],axis=0)
        return variant_ids, average_allele1_pred_counts, average_allele2_pred_counts, \
               average_allele1_pred_profiles, average_allele2_pred_profiles
    else:
        return variant_ids, allele1_pred_counts, allele2_pred_counts, \
               allele1_pred_profiles, allele2_pred_profiles

def get_variant_scores_with_peaks(allele1_pred_counts, allele2_pred_counts,
                       allele1_pred_profiles, allele2_pred_profiles, pred_counts):
    # logfc = np.log2(allele2_pred_counts / allele1_pred_counts)
    # jsd = np.array([jensenshannon(x,y,base=2.0) for x,y in zip(allele2_pred_profiles, allele1_pred_profiles)])

    logfc, jsd = get_variant_scores(allele1_pred_counts, allele2_pred_counts,
                                    allele1_pred_profiles, allele2_pred_profiles)
    allele1_quantile = np.array([np.max([np.mean(pred_counts < x), (1/len(pred_counts))]) for x in allele1_pred_counts])
    allele2_quantile = np.array([np.max([np.mean(pred_counts < x), (1/len(pred_counts))]) for x in allele2_pred_counts])

    return logfc, jsd, allele1_quantile, allele2_quantile

def get_variant_scores(allele1_pred_counts, allele2_pred_counts,
                       allele1_pred_profiles, allele2_pred_profiles):

    print('allele1_pred_counts shape:', allele1_pred_counts.shape)
    print('allele2_pred_counts shape:', allele2_pred_counts.shape)
    print('allele1_pred_profiles shape:', allele1_pred_profiles.shape)
    print('allele2_pred_profiles shape:', allele2_pred_profiles.shape)

    logfc = np.squeeze(np.log2(allele2_pred_counts / allele1_pred_counts))
    jsd = np.squeeze([jensenshannon(x, y, base=2.0)
                     for x,y in zip(softmax(allele2_pred_profiles),
                                    softmax(allele1_pred_profiles))])

    print('logfc shape:', logfc.shape)
    print('jsd shape:', jsd.shape)

    return logfc, jsd

def adjust_indel_jsd(variants_table,allele1_pred_profiles,allele2_pred_profiles,original_jsd):
    indel_idx = []
    for i, row in variants_table.iterrows():
        allele1, allele2 = row[['allele1','allele2']]
        if allele1 == "-":
            allele1 = ""
        if allele2 == "-":
            allele2 = ""
        if len(allele1) != len(allele2):
            indel_idx += [i]

    adjusted_jsd = []
    for i in indel_idx:
        row = variants_table.iloc[i]
        allele1, allele2 = row[['allele1','allele2']]
        if allele1 == "-":
            allele1 = ""
        if allele2 == "-":
            allele2 = ""

        allele1_length = len(allele1)
        allele2_length = len(allele2)

        allele1_p = allele1_pred_profiles[i]
        allele2_p = allele2_pred_profiles[i]
        assert len(allele1_p) == len(allele2_p)
        assert allele1_length != allele2_length
        flank_size = len(allele1_p)//2
        allele1_left_flank = allele1_p[:flank_size]
        allele2_left_flank = allele2_p[:flank_size]

        if allele1_length > allele2_length:
            allele1_right_flank = np.concatenate([allele1_p[flank_size:flank_size+allele2_length],allele1_p[flank_size+allele1_length:]])
            allele2_right_flank = allele2_p[flank_size:allele2_length-allele1_length]
        else:
            allele1_right_flank = allele1_p[flank_size:allele1_length-allele2_length]
            allele2_right_flank = np.concatenate([allele2_p[flank_size:flank_size+allele1_length], allele2_p[flank_size+allele2_length:]])


        adjusted_allele1_p = np.concatenate([allele1_left_flank,allele1_right_flank])
        adjusted_allele2_p = np.concatenate([allele2_left_flank,allele2_right_flank])
        adjusted_allele1_p = adjusted_allele1_p/np.sum(adjusted_allele1_p)
        adjusted_allele2_p = adjusted_allele2_p/np.sum(adjusted_allele2_p)
        assert len(adjusted_allele1_p) == len(adjusted_allele2_p)
        adjusted_j = jensenshannon(adjusted_allele1_p,adjusted_allele2_p,base=2.0)
        adjusted_jsd += [adjusted_j]

    adjusted_jsd_list = original_jsd.copy()
    if len(indel_idx) > 0:
        for i in range(len(indel_idx)):
            idx = indel_idx[i]
            adjusted_jsd_list[idx] = adjusted_jsd[i]

    return indel_idx, adjusted_jsd_list

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

def create_shuffle_table(variants_table, random_seed=None, total_shuf=None, num_shuf=None):
    if total_shuf != None:
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
        if num_shuf != None:
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

def get_pvals(obs, bg, tail):
    sorted_bg = np.sort(bg)
    if tail == 'right' or tail == 'both':
        rank_right = len(sorted_bg) - np.searchsorted(sorted_bg, obs, side='left')
        pval_right = (rank_right + 1) / (len(sorted_bg) + 1)
        if tail == 'right':
            return pval_right
    if tail == 'left' or tail == 'both':
        rank_left = np.searchsorted(sorted_bg, obs, side='right')
        pval_left = (rank_left + 1) / (len(sorted_bg) + 1)
        if tail == 'left':
            return pval_left
    assert tail == 'both'
    min_pval = np.minimum(pval_left, pval_right)
    pval_both = min_pval * 2

    return pval_both

def get_pvals_with_binomtest(ref_counts, alt_counts):
    pvals = []
    for ref_count, alt_count in zip(ref_counts, alt_counts):
        pval = binomtest(int(ceil(alt_count)), int(ceil(ref_count)) + int(ceil(alt_count)), 0.5).pvalue
        pvals.append(pval)
    return pvals

def geo_mean_overflow(iterable,axis=0):
    return np.exp(np.log(iterable).mean(axis=0))

def add_missing_columns_to_peaks_df(peaks, schema):
    if schema != 'narrowpeak':
        raise ValueError("Schema not supported")
    
    required_columns = get_peak_schema(schema)
    num_current_columns = peaks.shape[1]
    
    if num_current_columns == 10:
        peaks.columns = required_columns[:num_current_columns]
        return peaks  # No missing columns, return as is

    elif num_current_columns < 3:
        raise ValueError("Peaks dataframe has fewer than 3 columns, which is invalid")
    
    elif num_current_columns > 10:
        raise ValueError("Peaks dataframe has greater than 10 columns, which is invalid")
    
    # Add missing columns to reach a total of 10 columns
    peaks.columns = required_columns[:num_current_columns]
    columns_to_add = required_columns[num_current_columns:]
    
    for column in columns_to_add:
        peaks[column] = '.'
    
    # Calculate the summit column
    peaks['summit'] = (peaks['end'] - peaks['start']) // 2
    
    return peaks
