import argparse


def update_scoring_args(parser):
    parser.add_argument("-l", "--list", type=str, required=True, help="Path to TSV file containing a list of variants to score")
    parser.add_argument("-g", "--genome", type=str, required=True, help="Path to the genome FASTA")
    parser.add_argument("-pg", "--peak_genome", type=str, help="Path to the genome FASTA for peaks")
    parser.add_argument("-m", "--model", type=str, required=True, help="Path to the ChromBPNet model .h5 file to use for variant scoring. For most use cases, this should be the bias-corrected model (chrombpnet_nobias.h5)")
    parser.add_argument("-o", "--out_prefix", type=str, required=True, help="Output prefix for storing SNP effect score predictions from the script, in the form of <path>/<prefix>. Directory should already exist.")
    parser.add_argument("-s", "--chrom_sizes", type=str, required=True, help="Path to TSV file with chromosome sizes")
    parser.add_argument("-ps", "--peak_chrom_sizes", type=str, help="Path to TSV file with chromosome sizes for peak genome")
    parser.add_argument("-b", "--bias", type=str, help="Bias model to use for variant scoring")
    parser.add_argument("-li", "--lite", action='store_true', help="Models were trained with chrombpnet-lite")
    parser.add_argument("-dm", "--debug_mode", action='store_true', help="Display allele input sequences")
    parser.add_argument("-bs", "--batch_size", type=int, default=512, help="Batch size to use for the model")
    parser.add_argument("-sc", "--schema", type=str, choices=['bed', 'plink', 'chrombpnet', 'original'], default='chrombpnet', help="Format for the input variants TSV file")
    parser.add_argument("-p", "--peaks", type=str, help="Path to BED file containing peak regions")
    parser.add_argument("-n", "--num_shuf", type=int, default=10, help="Number of shuffled scores per SNP")
    parser.add_argument("-t", "--total_shuf", type=int, help="Total number of shuffled scores across all SNPs. Overrides --num_shuf")
    parser.add_argument("-mp", "--max_peaks", type=int, help="Maximum number of peaks to use for peak percentile calculation")
    parser.add_argument("-c", "--chrom", type=str, help="Only score SNPs in selected chromosome")
    parser.add_argument("-r", "--random_seed", type=int, default=1234, help="Random seed for reproducibility when sampling")
    parser.add_argument("--no_hdf5", action='store_true', help="Do not save detailed predictions in hdf5 file")
    parser.add_argument("-nc", "--num_chunks", type=int, default=10, help="Number of chunks to divide SNP file into")
    parser.add_argument("-fo", "--forward_only", action='store_true', help="Run variant scoring only on forward sequence. Default: False")
    parser.add_argument("-st", "--shap_type",  nargs='+', default=["counts"], help="ChromBPNet output for which SHAP values should be computed ('counts' or 'profile'). Default is 'counts'")
    parser.add_argument("-sh", "--shuffled_scores", type=str, help="Path to pre-computed shuffled scores")
    parser.add_argument("--merge", action='store_true', help="For per-chromosome scoring, merge all per-chromosome predictions into a single file, and deletes the per-chromosome files. Default is False.")

def fetch_scoring_args():
    parser = argparse.ArgumentParser()
    update_scoring_args(parser)
    args = parser.parse_args()
    print(args)
    return args

def update_shap_args(parser):
    parser.add_argument("-l", "--list", type=str, required=True, help="A TSV file containing a list of variants to score")
    parser.add_argument("-g", "--genome", type=str, required=True, help="Path to genome FASTA")
    parser.add_argument("-m", "--model", type=str, required=True, help="Path to the ChromBPNet model .h5 file to use for variant scoring. For most use cases, this should be the bias-corrected model (chrombpnet_nobias.h5)")
    parser.add_argument("-o", "--out_prefix", type=str, required=True, help="Output prefix for storing SNP effect score predictions from the script, in the form of <path>/<prefix>. Directory should already exist.")
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
    parser.add_argument("-sd", "--score_dir", type=str, required=True, help="Path to directory containing variant scores that will be used to generate summary")
    parser.add_argument("-sl", "--score_list",  nargs='+', required=True, help="Space-separated list of variant score file names that will be used to generate summary")
    parser.add_argument("-o", "--out_prefix", type=str, required=True, help="Output prefix for storing the summary file with average scores across folds, in the form of <path>/<prefix>. Directory should already exist.")
    parser.add_argument("-sc", "--schema", type=str, required=True, choices=['bed', 'plink', 'plink2', 'chrombpnet', 'original'], default='chrombpnet', help="Format for the input variants list")

def fetch_variant_summary_args():
    parser = argparse.ArgumentParser()
    update_variant_summary_args(parser)
    args = parser.parse_args()
    print(args)
    return args

def update_variant_annotation_args(parser):
    parser.add_argument(
        "-l", "--list", type=str, required=True,
        help=(
            "Path to TSV file containing the variant scores (or summarized scores) to annotate.\n"
            "Alternatively, provide a BED file of variants with --schema bed.\n"
            "The file should contain variant information compatible with the selected schema."
        )
    )
    parser.add_argument("-o", "--out_prefix", type=str, required=True, help="Output prefix for storing the annotated file, in the form of <path>/<prefix>. Directory should already exist.")
    parser.add_argument("-p", "--peaks", type=str, help="Path to BED file containing peak regions")
    parser.add_argument("--hits", type=str, help="Path to BED file containing motif hits regions")
    parser.add_argument("-ge", "--genes", type=str, help="Path to BED file containing gene regions")
    parser.add_argument("-sc", "--schema", type=str, required=False, choices=['bed', 'plink', 'plink2', 'chrombpnet', 'original'], default='chrombpnet', help="Format for the input variants list")

def fetch_variant_annotation_args():
    parser = argparse.ArgumentParser()
    update_variant_annotation_args(parser)
    args = parser.parse_args()
    print(args)

    # Assert that at least one of genes, peaks, or hits is provided
    if not args.genes and not args.peaks and not args.hits:
        print("Error: At least one of --genes, --peaks, or --hits must be provided for annotation.")
        parser.print_help()
        exit(1)

    return args
