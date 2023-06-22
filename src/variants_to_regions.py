import pandas as pd
import argparse
from generators.variant_generator import VariantGenerator
from utils.helpers import load_variant_table, get_valid_variants
from pathlib import Path
import warnings

def fetch_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--list",
        type=str,
        required=True,
        help="TSV file containing the list of variants",
    )
    parser.add_argument(
        "-sc",
        "--schema",
        type=str,
        choices=["bed", "plink", "neuro-variants", "chrombpnet", "original"],
        default="chrombpnet",
        help="Format for the input variants list",
    )
    parser.add_argument("-g", "--genome", type=str, required=True, help="Genome fasta")
    parser.add_argument(
        "-s",
        "--chrom_sizes",
        type=str,
        required=True,
        help="Path to TSV file with chromosome sizes",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Path to store shap summary h5 files, \
         directory should already exist",
    )
    parser.add_argument(
        "-il",
        "--input_len",
        type=int,
        default=2114,
        help="Length of the input sequence (in bp)",
    )
    args = parser.parse_args()
    return args


def main():
    args = fetch_args()
    print(args)

    if not Path(args.output_dir).exists():
        raise ValueError("Output directory does not exist")

    chrom_sizes = pd.read_csv(
        args.chrom_sizes, header=None, sep="\t", names=["chrom", "size"]
    )
    chrom_sizes_dict = chrom_sizes.set_index("chrom")["size"].to_dict()

    variants_table = load_variant_table(args.list, args.schema)
    variants_table = variants_table.fillna("-")
    print(f"Variants table shape before validation: {variants_table.shape}")
    variants_table = variants_table.loc[
        variants_table.apply(
            lambda x: get_valid_variants(
                x.chr, x.pos, x.allele1, x.allele2, args.input_len, chrom_sizes_dict
            ),
            axis=1,
        )
    ]
    variants_table.reset_index(drop=True, inplace=True)
    print(f"Variants table shape after validation: {variants_table.shape}")

    var_gen = VariantGenerator(
        variants_table=variants_table,
        input_len=args.input_len,
        genome_fasta=args.genome,
        batch_size=variants_table.shape[0],
        return_coords=True,
    )

    assert len(var_gen) == 1
    _, _, _, allele1_coords, allele2_coords = var_gen[0]
    # add empty columns and the last column
    empty_cols = ["."] * 6
    allele1_coords_all = [x + empty_cols + [args.input_len//2] for x in allele1_coords]
    allele2_coords_all = [x + empty_cols + [args.input_len//2] for x in allele2_coords]
    # make dataframes
    df1 = pd.DataFrame(allele1_coords_all)
    df2 = pd.DataFrame(allele2_coords_all)
    
    # filter out any indel variants (i.e. those with -1 in the position columns)
    # first detect any such variants and raise a warning
    if -1 in df1[1].to_list():
        msg = "It appears that you have indel variants. We don't support mapping these to regions " \
            "in the reference genome. Thus these won't be present in the resulting bed files."
        warnings.warn(msg)
    df1 = df1[df1[1] != -1]
    df2 = df2[df2[1] != -1]

    # df2 and df1 are expected to be equal since there are no indel's
    assert df1.equals(df2)

    # write to bed files
    df1.to_csv(Path(args.output_dir) / "variants_regions.bed", sep="\t", header=None, index=None)

if __name__ == "__main__":
    main()
