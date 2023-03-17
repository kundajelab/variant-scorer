import argparse

def update_scoring_args(parser):
    parser.add_argument("-l", "--list", type=str, required=True, help="TSV file containing list of variants to score")
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
    parser.add_argument("-sc", "--schema", type=str, choices=['bed', 'plink', 'neuro-variants', 'chrombpnet', 'original'], default='chrombpnet', help="Format for the input variants list")
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

