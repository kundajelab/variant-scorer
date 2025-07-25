# variant-scorer

The variant scoring repository provides a set of scripts for scoring genetic variants using a ChromBPNet model.

**Important notes:**

- in the input variant list, the `pos` (position) column is expected to be the 1-indexed SNP position, unless the schema is *bed*
- the reported log fold-change (`logFC`) for predicted variant effects is in log base 2
- by default, counts and profile prediction for each allele are averaged between the predictions obtained using
the forward sequence and the reverse-complement of that sequence as input. This can be disabled using
the `--forward_only` option to only use the forward sequence predictions (See [issue28](https://github.com/kundajelab/variant-scorer/issues/28#issuecomment-2900574336)
for a discussion).


# Variant inputs

## Variant file schemas

Variant lists should be provided as TSVs with one variant per row, and column
names adhering to one of the following schemas:

* chrombpnet : `['chr', 'pos', 'allele1', 'allele2', 'variant_id']`
* bed : `['chr', 'pos', 'end', 'allele1', 'allele2', 'variant_id']`
* plink : `['chr', 'variant_id', 'ignore1', 'pos', 'allele1', 'allele2']`
* original : `['chr', 'pos', 'variant_id', 'allele1', 'allele2']`

**NOTE:** The `pos` (position) column is expected to correspond to the 1-indexed variant position, unless the schema is `bed`.

## Specifying variants

For single-nucleotide variants, `allele1` and `allele2`, provide the corresponding nucleotide for each allele,
e.g. (for variants provided in the `chrombpnet` schema):

```
chr1	866281	C	T	1_866281_C_T
```

For deletions, use `-` for `allele2` to represent the deleted nucleotides, e.g.

```
chr1	866281	C	-	1_866281_Cdel
```

For insertions, use `-` for `allele1` to represent the inserted nucleotides, e.g.

```
chr1	866281	-	CT	1_866281_CTins
```



# Workflow

## 1. Score variants: `variant_scoring.py`

This script takes a list of variants in various input formats and generates scores
for the variants using a ChromBPNet model. The output is a TSV file containing the scores for each variant. 
Since variants are stored in memory, we also provide `variant_scoring.per_chrom.py` to score variants on a per-chromosome basis,
and write the scores per chromosome to file before proceeding to the next chromosome. Per-chromosome
files can then be merged automatically using the `--merge` option.

### Usage:


```bash
python src/variant_scoring.py --list [VARIANTS_FILE] \
    --genome [GENOME_FASTA] \
	--model [MODEL_PATH] \
	--out_prefix [OUT_PREFIX] \
	--chrom_sizes [CHROM_SIZES] \
	[OTHER_ARGS]
```

### Input arguments:


- `-h`, `--help`: Show help message with arguments and their descriptions, and exit
- `-l`, `--list` (**required**): Path to TSV file containing a list of variants to score
- `-g`, `--genome` (**required**): Path to the genome FASTA
- `-pg`, `--peak_genome`: Path to the genome FASTA for peaks
- `-m`, `--model` (**required**): Path to the ChromBPNet model .h5 file to use for variant scoring. For most use cases, this should be the bias-corrected model (chrombpnet_nobias.h5)
- `-o`, `--out_prefix` (**required**): Output prefix for storing SNP effect score predictions from the script, in the form of `<path>/<prefix>`. Directory should already exist.
- `-s`, `--chrom_sizes` (**required**): Path to TSV file with chromosome sizes
- `--no_hdf5`: Do not save basepair resolution predictions to hdf5 file. Recommended for large variant lists.
- `-ps`, `--peak_chrom_sizes`: Path to TSV file with chromosome sizes for peak genome
- `-b`, `--bias`: Bias model to use for variant scoring
- `-li`, `--lite`: Models were trained with chrombpnet-lite
- `-dm`, `--debug_mode`: Display allele input sequences
- `-bs`, `--batch_size`: Batch size to use for the model
- `-sc`, `--schema`: Format for the input variants TSV file. Choices: `bed`, `plink`, `chrombpnet`, `original`
- `-p`, `--peaks`: Path to BED file containing peak regions
- `-n`, `--num_shuf`: Number of shuffled scores per SNP
- `-t`, `--total_shuf`: Total number of shuffled scores across all SNPs. Overrides `--num_shuf`
- `-mp`, `--max_peaks`: Maximum number of peaks to use for peak percentile calculation
- `-c`, `--chrom`: Only score SNPs in selected chromosome
- `-r`, `--random_seed`: Random seed for reproducibility when sampling
- `--no_hdf5`: Do not save detailed predictions in hdf5 file
- `-nc`, `--num_chunks`: Number of chunks to divide SNP file into
- `-fo`, `--forward_only`: Run variant scoring only on forward sequence (Default: False)
- `-st`, `--shap_type`: ChromBPNet output for which SHAP values should be computed (`counts` or `profile`). Default is `counts`
- `-sh`, `--shuffled_scores`: Path to pre-computed shuffled scores



### Outputs:

The variant scores are stored in `<out_prefix>.variant_scores.tsv`.

Predicted effects are computed as `allele2` vs `allele1`. For each variant, we 
compute the following metrics, as described in the [ChromPBNet preprint](https://www.biorxiv.org/content/10.1101/2024.12.25.630221v1.full.pdf+html):

- `logfc`: Log fold-change of total predicted coverage for `allele2` vs `allele1`, providing a canonical effect size of the variant on local accessibility. A higher `logFC` indicates higher predicted accessibility for `allele2` compared to `allele1`.
- `abs_logfc`: Absolute value of the log fold-change.
- `active_allele_quantile`: Active Allele Quantile is the percentile of the predicted total coverage of the stronger allele relative to the distribution of predicted total coverage across all ATAC-seq/DNase-seq peaks.  
- `jsd`: Jensen-Shannon distance between the bias-corrected base-resolution probability profiles of the two alleles, which captures effects on profile shape, such as changes in TF footprints.

We provide several additional metrics that are computed as the product of the above metrics:

- `abs_logfc_x_jsd`: described in the preprint as Integrative  Effect  Size  (IES), the  product  of  logFC  and  JSD,
- `logfc_x_active_allele_quantile`
- `abs_logfc_x_active_allele_quantile`
- `jsd_x_active_allele_quantile`
- `logfc_x_jsd_x_active_allele_quantile`: described in the preprint as Integrative Prioritization Score (IPS) is the product of logFC, JSD, and AAQ
- `abs_logfc_x_jsd_x_active_allele_quantile`

*__NOTE__*: For profile predictions, the saved arrays consist of model logits, not probabilities. This allows for averaging profile predictions across folds more easily, by averaging logits over folds and then taking the softmax (see [`variant-scorer/pull/23`](https://github.com/kundajelab/variant-scorer/pull/23)).




## 2. Summarize variant scores across model folds: `variant_summary_across_folds.py`

This script takes variant scores generated by the `variant_scoring.py` script for several model folds,
and generates a TSV file with the mean scores for each score metric across folds.

### Usage:

```bash
python src/variant_summary_across_folds.py \
	--score_dir [VARIANT_SCORE_DIR] \
	--score_list [SCORE_LIST] \
	--out_prefix [OUT_PREFIX] \
	--schema [SCHEMA]
```

### Input arguments:


- `-h`, `--help`: Show help message with arguments and their descriptions, and exit
- `-sd`, `--score_dir` (**required**): Path to directory containing variant scores that will be used to generate summary
- `-sl`, `--score_list` (**required**): Space-separated list of variant score file names that will be used to generate summary. Files should exist in `--score_dir`.
- `-o`, `--out_prefix` (**required**): Output prefix for storing the summary file with average scores across folds, in the form of `<path>/<prefix>`. Directory should already exist.
- `-sc`, `--schema`: Format for the input variants list. Choices: `bed`, `plink`, `plink2`, `chrombpnet`, `original`. Default is `chrombpnet`.


### Outputs:

The summary file is stored at `<out_prefix>.mean.variant_scores.tsv`.


## 3. Annotate variants: `variant_annotation.py`

This script takes a list of variants and annotates each with their closest genes,
and/or overlaps with peaks or motif hits.

**NOTE:** This script assumes that the genes, peaks, and hits are in the same reference genome as the variants, and it does not perform any liftOver operations.

### Usage:

```bash
python src/variant_annotation.py \
	--list [VARIANT_SCORES or VARIANT_LIST] \
	--out_prefix [OUT_PREFIX] \
	--peaks [PEAKS] \
	--genes [GENES] \
	--hits [HITS] \
	--schema [SCHEMA]
```

### Input arguments:


- `-h`, `--help`: Show help message with arguments and their descriptions, and exit
- `-l`, `--list` (**required**): Path to TSV file containing scored variants as output by `variant_scoring.py` (or the summary across folds), or a BED file of variants with the `--schema bed` option.
- `-o`, `--out_prefix` (**required**): Output prefix for storing the annotated file, in the form of `<path>/<prefix>`. Directory should already exist.
- `-sc`, `--schema`: Format for the input variants list. Use `bed` if providing BED file of variants.
- `-ge`, `--genes`: Path to BED file containing gene regions. If provided, the script will annotate each variant with the three closest genes and the distance to each.
- `-p`, `--peaks`: Path to BED file containing peak regions. If provided, the script will annotate each variant according to whether it overlaps with any peak.
- `--hits`: Path to BED file containing motif hits, with columns `chr`, `start`, `end`, `motif`, `score`, `strand`, `class`. If provided, the script will annotate variants with whether they overlap any motif hits (and which they overlap).

At least one of `--genes`, `--peaks`, or `--hits` must be provided for annotation.

## 4. Compute variant SHAP scores: `variant_shap.py`

This script computes the contribution scores for each variant, for allele1
and allele2, with respect to the specified ChromBPNet model output (`counts` or `profile`).

```bash
python src/variant_shap.py \
	--list [VARIANTS_FILE] \
	--genome [GENOME] \
	--chrom_sizes [CHROM_SIZES] \
	--model [MODEL_PATH] \
	--out_prefix [OUT_PREFIX] \
	--schema [SCHEMA] \
	--shap_type [SHAP_TYPE] \
	[OTHER_ARGS]
```

### Input arguments:

- `-h`, `--help`: Show help message with arguments and their descriptions, and exit
- `-l`, `--list` (**required**): A TSV file containing a list of variants to score
- `-g`, `--genome` (**required**): Path to genome FASTA
- `-m`, `--model` (**required**): Path to the ChromBPNet model .h5 file to use for variant scoring. For most use cases, this should be the bias-corrected model (chrombpnet_nobias.h5)
- `-o`, `--out_prefix` (**required**): Output prefix for storing SNP effect score predictions from the script, in the form of `<path>/<prefix>`. Directory should already exist.
- `-s`, `--chrom_sizes` (**required**): Path to TSV file with chromosome sizes
- `-li`, `--lite`: Models were trained with chrombpnet-lite
- `-dm`, `--debug_mode`: Display allele input sequences
- `-bs`, `--batch_size`: Batch size to use for the model. Default is 10000
- `-sc`, `--schema`: Format for the input variants list. Choices: `bed`, `plink`, `chrombpnet`, `original`. Default is `chrombpnet`
- `-c`, `--chrom`: Only score SNPs in selected chromosome
- `-st`, `--shap_type`: ChromBPNet output for which SHAP values should be computed. Can specify multiple values. Default is `counts`


### Outputs:

The variant SHAP scores are stored in `<out_prefix>.variant_shap.<shap_type>.h5`.

The h5 file contains the following datasets:

- `alleles`: shape `(2 * number of variants, )`, binary array indicating whether the allele is allele1 (0) or allele2 (1)
- `raw/seq`: shape `(2 * number of variants, 4, 2114)`, contains one hot encoding of sequences around the variant which were scored
- `shap/seq`: shape `(2 * number of variants, 4, 2114)`, contains hypothetical contribution scores
- `projected_shap/seq`: shape `(2 * number of variants, 4, 2114)`, contains values obtained by multiplying hypothetical SHAP values with raw (one-hot encoded) sequences
- `variant_ids`: shape `(2 * number of variants, )`, contains variant identifiers corresponding to each variant as provided in the input list