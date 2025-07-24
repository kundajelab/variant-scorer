import pandas as pd
import numpy as np
import os
import argparse
import scipy.stats
from scipy.spatial.distance import jensenshannon
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import load_model
import tensorflow as tf
import h5py
import math
from generators.variant_generator import VariantGenerator
from generators.peak_generator import PeakGenerator
from utils import argmanager, losses
from utils.helpers import *
from utils.io import *
import shap
from utils.shap_utils import *
import deepdish as dd
tf.compat.v1.disable_v2_behavior()


def main():
    args = argmanager.fetch_shap_args()
    print(args)

    out_dir = os.path.sep.join(args.out_prefix.split(os.path.sep)[:-1])
    print()
    print('out_dir:', out_dir)
    print()
    if not os.path.exists(out_dir):
        raise OSError("Output directory does not exist")

    model = load_model_wrapper(args.model)
    variants_table = load_variant_table(args.list, args.schema)
    variants_table = variants_table.fillna('-')

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
    variants_table = variants_table.loc[variants_table.apply(lambda x: get_valid_variants(x.chr, x.pos, x.allele1, x.allele2, input_len, chrom_sizes_dict), axis=1)]
    variants_table.reset_index(drop=True, inplace=True)
    print(variants_table.shape)
    
    for shap_type in args.shap_type:
        # fetch model prediction for variants
        batch_size=args.batch_size
        ### set the batch size to the length of variant table in case variant table is small to avoid error
        batch_size=min(batch_size,len(variants_table))
        # output_file=h5py.File(''.join([args.out_prefix, ".variant_shap.%s.h5"%shap_type]), 'w')
        # observed = output_file.create_group('observed')
        # allele1_write = observed.create_dataset('allele1_shap', (len(variants_table),2114,4), chunks=(batch_size,2114,4), dtype=np.float16, compression='gzip', compression_opts=9)
        # allele2_write = observed.create_dataset('allele2_shap', (len(variants_table),2114,4), chunks=(batch_size,2114,4), dtype=np.float16, compression='gzip', compression_opts=9)
        # variant_ids_write = observed.create_dataset('variant_ids', (len(variants_table),), chunks=(batch_size,), dtype='S100', compression='gzip', compression_opts=9)

        allele1_seqs = []
        allele2_seqs = []
        allele1_scores = []
        allele2_scores = []
        variant_ids = []

        num_batches=len(variants_table)//batch_size
        for i in range(num_batches):
            sub_table=variants_table[i*batch_size:(i+1)*batch_size]
            var_ids, allele1_inputs, allele2_inputs, \
            allele1_shap, allele2_shap = fetch_shap(model,
                                                    sub_table,
                                                    input_len,
                                                    args.genome,
                                                    args.batch_size,
                                                    debug_mode=args.debug_mode,
                                                    lite=args.lite,
                                                    bias=None,
                                                    shuf=False,
                                                    shap_type=shap_type)
            
            # allele1_write[i*batch_size:(i+1)*batch_size] = allele1_shap
            # allele2_write[i*batch_size:(i+1)*batch_size] = allele2_shap
            # variant_ids_write[i*batch_size:(i+1)*batch_size] = [s.encode("utf-8") for s in var_ids]

            if len(variant_ids) == 0:
                allele1_seqs = allele1_inputs
                allele2_seqs = allele2_inputs
                allele1_scores = allele1_shap
                allele2_scores = allele2_shap
                variant_ids = var_ids
            else:
                allele1_seqs = np.concatenate((allele1_seqs, allele1_inputs))
                allele2_seqs = np.concatenate((allele2_seqs, allele2_inputs))
                allele1_scores = np.concatenate((allele1_scores, allele1_shap))
                allele2_scores = np.concatenate((allele2_scores, allele2_shap))
                variant_ids = np.concatenate((variant_ids, var_ids))

        if len(variants_table)%batch_size != 0:
            sub_table=variants_table[num_batches*batch_size:len(variants_table)]
            var_ids, allele1_inputs, allele2_inputs, \
            allele1_shap, allele2_shap = fetch_shap(model,
                                                                    sub_table,
                                                                    input_len,
                                                                    args.genome,
                                                                    args.batch_size,
                                                                    debug_mode=args.debug_mode,
                                                                    lite=args.lite,
                                                                    bias=None,
                                                                    shuf=False,
                                                                    shap_type=shap_type)
            
            # allele1_write[num_batches*batch_size:len(variants_table)] = allele1_shap
            # allele2_write[num_batches*batch_size:len(variants_table)] = allele2_shap
            # variant_ids_write[num_batches*batch_size:len(variants_table)] = [s.encode("utf-8") for s in var_ids]

            if len(variant_ids) == 0:
                allele1_seqs = allele1_inputs
                allele2_seqs = allele2_inputs
                allele1_scores = allele1_shap
                allele2_scores = allele2_shap
                variant_ids = var_ids
            else:
                allele1_seqs = np.concatenate((allele1_seqs, allele1_inputs))
                allele2_seqs = np.concatenate((allele2_seqs, allele2_inputs))
                allele1_scores = np.concatenate((allele1_scores, allele1_shap))
                allele2_scores = np.concatenate((allele2_scores, allele2_shap))
                variant_ids = np.concatenate((variant_ids, var_ids))

        # # store shap at variants
        # with h5py.File(''.join([args.out_prefix, ".variant_shap.%s.h5"%shap_type]), 'w') as f:
        #     observed = f.create_group('observed')
        #     observed.create_dataset('allele1_shap', data=allele1_shap, compression='gzip', compression_opts=9)
        #     observed.create_dataset('allele2_shap', data=allele2_shap, compression='gzip', compression_opts=9)
            
        assert(allele1_seqs.shape==allele1_scores.shape)
        assert(allele2_seqs.shape==allele2_scores.shape)
        assert(allele1_seqs.shape==allele2_seqs.shape)
        assert(allele1_scores.shape==allele2_scores.shape)
        assert(allele1_seqs.shape[2]==4)
        assert(len(allele1_seqs==len(variant_ids)))
        
        shap_dict = {
            'raw': {'seq': np.concatenate((np.transpose(allele1_seqs, (0, 2, 1)).astype(np.int8),
                                           np.transpose(allele2_seqs, (0, 2, 1)).astype(np.int8)))},
            'shap': {'seq': np.concatenate((np.transpose(allele1_scores, (0, 2, 1)).astype(np.float16),
                                            np.transpose(allele2_scores, (0, 2, 1)).astype(np.float16)))},
            'projected_shap': {'seq': np.concatenate((np.transpose(allele1_seqs * allele1_scores, (0, 2, 1)).astype(np.float16),
                                                      np.transpose(allele2_seqs * allele2_scores, (0, 2, 1)).astype(np.float16)))},
            'variant_ids': np.concatenate((np.array(variant_ids), np.array(variant_ids))),
            'alleles': np.concatenate((np.array([0] * len(variant_ids)),
                                       np.array([1] * len(variant_ids))))}

        dd.io.save(''.join([args.out_prefix, ".variant_shap.%s.h5"%shap_type]),
                   shap_dict,
                   compression='blosc')

    print("DONE")


if __name__ == "__main__":
    main()
