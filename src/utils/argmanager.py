import sys
import argparse


def update_scoring_args(parser):
    parser.add_argument("-l", "--list", type=str, required=True, help="a TSV file containing a list of variants to score")
    parser.add_argument("-g", "--genome", type=str, required=True, help="Genome fasta")
    parser.add_argument("-pg", "--peak_genome", type=str, help="Genome fasta for peaks")
    parser.add_argument("-m", "--model", type=str, required=True, help="ChromBPNet model to use for variant scoring")
    parser.add_argument("-o", "--out_prefix", type=str, required=True, help="Path to storing snp effect score predictions from the script, directory should already exist")
    parser.add_argument("-s", "--chrom_sizes", type=str, required=True, help="Path to TSV file with chromosome sizes")
    parser.add_argument("-ps", "--peak_chrom_sizes", type=str, help="Path to TSV file with chromosome sizes for peak genome")
    parser.add_argument("-b", "--bias", type=str, help="Bias model to use for variant scoring")
    parser.add_argument("-li", "--lite", action='store_true', help="Models were trained with chrombpnet-lite")
    parser.add_argument("-dm", "--debug_mode", action='store_true', help="Display allele input sequences")
    parser.add_argument("-bs", "--batch_size", type=int, default=512, help="Batch size to use for the model")
    parser.add_argument("-sc", "--schema", type=str, choices=['bed', 'plink', 'chrombpnet', 'original'], default='chrombpnet', help="Format for the input variants list")
    parser.add_argument("-p", "--peaks", type=str, help="Bed file containing peak regions")
    parser.add_argument("-n", "--num_shuf", type=int, default=10, help="Number of shuffled scores per SNP")
    parser.add_argument("-t", "--total_shuf", type=int, help="Total number of shuffled scores across all SNPs. Overrides --num_shuf")
    parser.add_argument("-mp", "--max_peaks", type=int, help="Maximum number of peaks to use for peak percentile calculation")
    parser.add_argument("-c", "--chrom", type=str, help="Only score SNPs in selected chromosome")
    parser.add_argument("-r", "--random_seed", type=int, default=1234, help="Random seed for reproducibility when sampling")
    parser.add_argument("--no_hdf5", action='store_true', help="Do not save detailed predictions in hdf5 file")
    parser.add_argument("-nc", "--num_chunks", type=int, default=10, help="Number of chunks to divide SNP file into")
    parser.add_argument("-fo", "--forward_only", action='store_true', help="Run variant scoring only on forward sequence")
    parser.add_argument("-st", "--shap_type",  nargs='+', default=["counts"])

def fetch_scoring_args():
    parser = argparse.ArgumentParser()
    update_scoring_args(parser)
    args = parser.parse_args()
    print(args)
    return args

def update_shap_args(parser):
    parser.add_argument("-l", "--list", type=str, required=True, help="a TSV file containing a list of variants to score")
    parser.add_argument("-g", "--genome", type=str, required=True, help="Genome fasta")
    parser.add_argument("-m", "--model", type=str, required=True, help="ChromBPNet model to use for variant scoring")
    parser.add_argument("-o", "--out_prefix", type=str, required=True, help="Path to storing snp effect score predictions from the script, directory should already exist")
    parser.add_argument("-s", "--chrom_sizes", type=str, required=True, help="Path to TSV file with chromosome sizes")
    parser.add_argument("-li", "--lite", action='store_true', help="Models were trained with chrombpnet-lite")
    parser.add_argument("-dm", "--debug_mode", action='store_true', help="Display allele input sequences")
    parser.add_argument("-bs", "--batch_size", type=int, default=10000, help="Batch size to use for the model")
    parser.add_argument("-sc", "--schema", type=str, choices=['bed', 'plink', 'chrombpnet', 'original'], default='chrombpnet', help="Format for the input variants list")
    parser.add_argument("-c", "--chrom", type=str, help="Only score SNPs in selected chromosome")
    parser.add_argument("-st", "--shap_type",  nargs='+', default=["counts"])
    
def fetch_shap_args():
    parser = argparse.ArgumentParser()
    update_shap_args(parser)
    args = parser.parse_args()
    print(args)
    return args

def update_variant_summary_args(parser):
    parser.add_argument("-sd", "--score_dir", type=str, required=True, help="Path to directory with variant scores that will be used to generate summary")
    parser.add_argument("-sl", "--score_list",  nargs='+', required=True, help="Names of variant score files that will be used to generate summary")
    parser.add_argument("-o", "--out_prefix", type=str, required=True, help="Path prefix for storing the summary file with average scores across folds; directory should already exist")
    parser.add_argument("-sc", "--schema", type=str, required=True, choices=['bed', 'plink', 'plink2', 'chrombpnet', 'original'], default='chrombpnet', help="Format for the input variants list")

def fetch_variant_summary_args():
    parser = argparse.ArgumentParser()
    update_variant_summary_args(parser)
    args = parser.parse_args()
    print(args)
    return args

def update_variant_annotation_args(parser):
    parser.add_argument("-l", "--list", type=str, required=True, help="a TSV file containing a list of variants to annotate.")
    parser.add_argument("-o", "--out_prefix", type=str, required=True, help="Path prefix for storing the annotated file; directory should already exist.")
    parser.add_argument("-p", "--peaks", type=str, help="Adds overlapping peaks information. Bed file containing peak regions.")
    parser.add_argument("-g", "--closest-genes", type=str, help="Adds closest gene annotations. Bed file containing gene regions. Default amount is 3.")
    parser.add_argument("-gc", "--closest-gene-count", type=int, help="Changes the number of closest genes (using the -g flag) annotated to the specified number.")
    parser.add_argument("-r2", "--r2", type=str, help="Adds r2 annotations. Requires a PLINK .ld file.")
    parser.add_argument("-sc", "--schema", type=str, required=True, choices=['bed', 'plink', 'plink2', 'chrombpnet', 'original'], default='chrombpnet', help="Format for the input variants list.")
    parser.add_argument("-v", "--verbose", help="Enable detailed logging.", action="store_true")

def fetch_variant_annotation_args():
    parser = argparse.ArgumentParser()
    update_variant_annotation_args(parser)
    args = parser.parse_args()
    print(args)
    return args

def fetch_main_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True, dest='subcommand', description="variant-scorer: a tool for scoring and annotating genetic variants.")

    scoring_parser = subparsers.add_parser('score', help='Score a list of variants using a model.')
    update_scoring_args(scoring_parser)

    shap_parser = subparsers.add_parser('shap', help='Calculate SHAP scores using a model.')
    update_shap_args(shap_parser)

    summary_parser = subparsers.add_parser('summary', help='Generate a summary file combining scores from multiple folds.')
    update_variant_summary_args(summary_parser)

    annotation_parser = subparsers.add_parser('annotate', help='Generate annotations for a list of variants.')
    update_variant_annotation_args(annotation_parser)

    # Check if a subcommand has been provided; if not, print help and exit
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    return args
