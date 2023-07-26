import pandas as pd
import numpy as np
import os

from utils.argmanager import *
from utils.helpers import *


def main():
    args = fetch_variant_summary_args()
    print(args)
    variant_score_dir = args.score_dir
    variant_table_list = args.score_list
    output_prefix = args.out_prefix

    score_dict = {}
    for i in range(len(variant_table_list)):
        variant_score_file = os.path.join(variant_score_dir, variant_table_list[i])
        assert os.path.isfile(variant_score_file)
        var_score = pd.read_table(variant_score_file)
        score_dict[i] = var_score

    variant_scores = score_dict[0][get_variant_schema(args.schema)].copy()
    for i in score_dict:
        assert score_dict[i]['chr'].tolist() == variant_scores['chr'].tolist()
        assert score_dict[i]['pos'].tolist() == variant_scores['pos'].tolist()
        assert score_dict[i]['allele1'].tolist() == variant_scores['allele1'].tolist()
        assert score_dict[i]['allele2'].tolist() == variant_scores['allele2'].tolist()
        assert score_dict[i]['variant_id'].tolist() == variant_scores['variant_id'].tolist()

    for score in ["logfc", "abs_logfc", "jsd", "logfc_x_jsd", "abs_logfc_x_jsd",
                  "logfc_x_max_percentile", "abs_logfc_x_max_percentile", "jsd_x_max_percentile",
                  "logfc_x_jsd_max_percentile", "abs_logfc_x_jsd_x_max_percentile"]:
        if score in score_dict[0]:
            variant_scores.loc[:, (score + '.mean')] = np.mean(np.array([score_dict[fold][score].tolist()
                                                                    for fold in score_dict]), axis=0)
            if score + '.pval' in score_dict[0]:
                variant_scores.loc[:, (score + '.mean' + '.pval')] = geo_mean_overflow([score_dict[fold][score + '.pval'].values for fold in score_dict])
            elif score + '_pval' in score_dict[0]:
                variant_scores.loc[:, (score + '.mean' + '.pval')] = geo_mean_overflow([score_dict[fold][score + '_pval'].values for fold in score_dict])

    print()
    print(variant_scores.head())
    print("Summary score table shape:", variant_scores.shape)
    print()

    out_file = output_prefix + ".mean.variant_scores.tsv"
    variant_scores.to_csv(out_file,\
                          sep="\t",\
                          index=False)

    print("DONE")


if __name__ == "__main__":
    main()
