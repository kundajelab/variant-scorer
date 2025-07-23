import pandas as pd
import pybedtools
from utils.argmanager import *
from utils.helpers import *
pd.set_option('display.max_columns', 20)


def main():
    args = fetch_variant_annotation_args()
    print(args)
    variant_scores_file = args.list
    output_prefix = args.out_prefix
    peak_path = args.peaks
    hits_path = args.hits
    genes = args.genes

    print("Loading variant scores from:", variant_scores_file)
    variant_scores = pd.read_table(variant_scores_file)

    if args.schema == "bed":
        if variant_scores['pos'].equals(variant_scores['end']):
            variant_scores['pos'] = variant_scores['pos'] - 1
        variant_scores_bed_format = variant_scores[['chr','pos','end','allele1','allele2','variant_id']].copy()
        variant_scores_bed_format.sort_values(by=["chr","pos","end"], inplace=True)
    else:
        # otherwise, assuming output of variant scoring pipeline.
        # convert to bed format
        try:
            variant_scores_bed_format = variant_scores[['chr','pos','allele1','allele2','variant_id']].copy()
        except KeyError:
            print("Error: The input file does not contain the required columns:" \
                "'chr', 'pos', 'allele1', 'allele2', 'variant_id'. " \
                "Please provide scores output of variant_scoring.py (or summarized scores) as input.")
            return
        variant_scores_bed_format['pos']  = variant_scores_bed_format.apply(lambda x: int(x.pos)-1, axis = 1)
        variant_scores_bed_format['end']  = variant_scores_bed_format.apply(lambda x: int(x.pos)+len(x.allele1), axis = 1)
        variant_scores_bed_format = variant_scores_bed_format[['chr','pos','end','allele1','allele2','variant_id']]
        variant_scores_bed_format.sort_values(by=["chr","pos","end"], inplace=True)

    print()
    print(variant_scores_bed_format.head())
    print("Variants table shape:", variant_scores_bed_format.shape)
    print()

    variant_bed = pybedtools.BedTool.from_dataframe(variant_scores_bed_format)

    # Process overlaps between variants and provided genes
    if args.genes:
        print("annotating with closest genes")
        gene_df = pd.read_table(genes, header=None)
        gene_bed = pybedtools.BedTool.from_dataframe(gene_df)
        closest_genes_bed = variant_bed.closest(gene_bed, d=True, t='first', k=3)

        closest_gene_df = closest_genes_bed.to_dataframe(header=None)

        print()
        print(closest_gene_df.head())
        print("Closest genes table shape:", closest_gene_df.shape)
        print()

        closest_genes = {}
        gene_dists = {}

        for index, row in closest_gene_df.iterrows():
            if not row[5] in closest_genes:
                closest_genes[row[5]] = []
                gene_dists[row[5]] = []
            closest_genes[row[5]].append(row.iloc[9])
            gene_dists[row[5]].append(row.iloc[-1])

        closest_gene_df = closest_gene_df.rename({5: 'variant_id'}, axis=1)
        closest_gene_df = closest_gene_df[['variant_id']]
        closest_gene_df['closest_gene_1'] = closest_gene_df['variant_id'].apply(lambda x: closest_genes[x][0] if len(closest_genes[x]) > 0 else '.')
        closest_gene_df['gene_distance_1'] = closest_gene_df['variant_id'].apply(lambda x: gene_dists[x][0] if len(closest_genes[x]) > 0 else '.')

        closest_gene_df['closest_gene_2'] = closest_gene_df['variant_id'].apply(lambda x: closest_genes[x][1] if len(closest_genes[x]) > 1 else '.')
        closest_gene_df['gene_distance_2'] = closest_gene_df['variant_id'].apply(lambda x: gene_dists[x][1] if len(closest_genes[x]) > 1 else '.')

        closest_gene_df['closest_gene_3'] = closest_gene_df['variant_id'].apply(lambda x: closest_genes[x][2] if len(closest_genes[x]) > 2 else '.')
        closest_gene_df['gene_distance_3'] = closest_gene_df['variant_id'].apply(lambda x: gene_dists[x][2] if len(closest_genes[x]) > 2 else '.')

        closest_gene_df = closest_gene_df[['variant_id', 'closest_gene_1', 'gene_distance_1',
                                           'closest_gene_2', 'gene_distance_2',
                                           'closest_gene_3', 'gene_distance_3']]
        closest_gene_df.drop_duplicates(inplace=True)
        variant_scores = variant_scores.merge(closest_gene_df, on='variant_id', how='left')

    # Process overlaps between variants and provided peak regions
    if args.peaks:
        print("annotating with peak overlap")
        peak_df = pd.read_table(peak_path, header=None)
        print(peak_df.head())
        peak_bed = pybedtools.BedTool.from_dataframe(peak_df)
        peak_intersect_bed = variant_bed.intersect(peak_bed, wa=True, u=True)

        peak_intersect_df = peak_intersect_bed.to_dataframe(names=variant_scores_bed_format.columns.tolist())

        print()
        print(peak_intersect_df.head())
        print("Peak overlap table shape:", peak_intersect_df.shape)
        print()

        # If non-empty
        if not peak_intersect_df.empty:
            variant_scores['peak_overlap'] = variant_scores['variant_id'].isin(peak_intersect_df['variant_id'].tolist())

    # Process overlaps between variants and provided motif hits
    if args.hits:
        print("annotating with motif hits overlap")
        hits_df = pd.read_table(hits_path, header=None)

        # set column names 
        hits_df.columns = ['chr_hit', 'start_hit', 'end_hit', 'motif', 'score', 'strand', 'class']

        print(hits_df.head())
        hits_bed = pybedtools.BedTool.from_dataframe(hits_df)
        hits_intersect_bed = variant_bed.intersect(hits_bed, wo=True)
        print(hits_intersect_bed.head())

        hits_intersect_df = hits_intersect_bed.to_dataframe(names=variant_scores_bed_format.columns.tolist() + hits_df.columns.tolist() + ['overlap_length'])

        print()
        print("Motif hits overlap table shape:", hits_intersect_df.shape)
        print()

        # If non-empty
        if not hits_intersect_df.empty:
            print(hits_intersect_df.head())

            # Make a boolean column indicating if the variant overlaps with motif hits
            variant_scores['hits_overlap'] = variant_scores['variant_id'].isin(hits_intersect_df['variant_id'].tolist())

            # Collapse the list of motif names for each variant
            hits_intersect_df['hits_motifs'] = hits_intersect_df.groupby('variant_id')['motif'].transform(lambda x: ','.join(set(x)))
            hits_intersect_df = hits_intersect_df[['variant_id', 'hits_motifs']].drop_duplicates()
            variant_scores = variant_scores.merge(hits_intersect_df, on='variant_id', how='left')
            variant_scores['hits_motifs'] = variant_scores['hits_motifs'].fillna('-')

        else:
            # add empty column if no hits overlap found
            variant_scores['hits_overlap'] = False
            variant_scores['hits_motifs'] = '-'
            print("No motif hits overlap found.")

    print()
    print(variant_scores.head())
    print("Annotation table shape:", variant_scores.shape)
    print()

    # Print some summary statistics:
    if args.peaks:
        print("Number of variants overlapping peaks:", variant_scores['peak_overlap'].sum(), "/", variant_scores.shape[0])
    
    if args.hits:
        print("Number of variants overlapping motif hits:", variant_scores['hits_overlap'].sum(), "/", variant_scores.shape[0])

    out_file = output_prefix + ".annotations.tsv"
    variant_scores.to_csv(out_file, sep="\t", index=False)

    print("DONE")
    print()


if __name__ == "__main__":
    main()
