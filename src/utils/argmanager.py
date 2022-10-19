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
    parser.add_argument("-sc", "--schema", type=str, choices=['bed', 'plink', 'chrombpnet'], default='chrombpnet', help="Format for the input variants list")
    parser.add_argument("-p", "--peaks", type=str, help="Bed file containing peak regions")
    parser.add_argument("-n", "--num_shuf", type=int, default=10, help="Number of shuffled scores for each SNP")
    parser.add_argument("-c", "--chrom", type=str, help="Only score SNPs in selected chromosome")

def fetch_scoring_args():
    parser = argparse.ArgumentParser()
    update_scoring_args(parser)
    args = parser.parse_args()
    print(args)
    return args

