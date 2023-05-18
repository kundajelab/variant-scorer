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

    input_len = model.input_shape[1]

    upstream_flank = args.upstream_flank
    downstream_flank = args.downstream_flank

    print(upstream_flank)
    if upstream_flank != None:
        assert downstream_flank != None
        input_len = input_len - len(downstream_flank) - len(upstream_flank)

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

        shuf_rsids, shuf_allele1_pred_counts, shuf_allele2_pred_counts = fetch_variant_predictions(model,
                                                                            shuf_variants_table,
                                                                            input_len,
                                                                            args.genome,
                                                                            args.batch_size,
                                                                            debug_mode=args.debug_mode,
                                                                            shuf=True,
                                                                            forward_only=args.forward_only,
                                                                            downstream_flank=downstream_flank,
                                                                            upstream_flank=upstream_flank)
        shuf_logfc = get_variant_scores(shuf_allele1_pred_counts,
                                                shuf_allele2_pred_counts)
        
        shuf_abs_logfc = np.squeeze(np.abs(shuf_logfc))

    if args.debug_mode:
        variants_table = variants_table.sample(10000, random_state=args.random_seed, ignore_index=True)
        print()
        print(variants_table.head())
        print("Debug variants table shape:", variants_table.shape)
        print()

    # fetch model prediction for variants
    rsids, allele1_pred_counts, allele2_pred_counts = fetch_variant_predictions(model,
                                                                        variants_table,
                                                                        input_len,
                                                                        args.genome,
                                                                        args.batch_size,
                                                                        debug_mode=args.debug_mode,
                                                                        shuf=False,
                                                                        forward_only=args.forward_only,
                                                                        downstream_flank=downstream_flank,
                                                                        upstream_flank=upstream_flank)

    variant_score = get_variant_scores(allele1_pred_counts,
                                    allele2_pred_counts)

    # unpack rsids to write outputs and write score to output
    assert np.array_equal(variants_table["rsid"].tolist(), rsids)
    variants_table["allele1_pred_LFC"] = allele1_pred_counts
    variants_table["allele2_pred_LFC"] = allele2_pred_counts
    variants_table["variant_score"] = variant_score
    variants_table["abs_variant_score"] = abs(variants_table["variant_score"])

    if len(shuf_variants_table) > 0:
        variants_table["abs_variant_score_pval"] = get_pvals(variants_table["abs_variant_score"].tolist(), shuf_abs_logfc)

    print()
    print(variants_table.head())
    print("Output score table shape:", variants_table.shape)
    print()

    variants_table.to_csv('.'.join([args.out_prefix, "variant_scores.tsv"]), sep="\t", index=False)
    print("DONE")
    print()

if __name__ == "__main__":
    main()
