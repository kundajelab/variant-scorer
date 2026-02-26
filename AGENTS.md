# AGENTS.md — variant-scorer

This file is the entry point for any agent working in this codebase. Read it fully before writing or running any code.


## Project Overview

The `variant-scorer` repository provides a suite of tools to predict the functional impact of genetic variants (SNPs, insertions, deletions) using **ChromBPNet** deep learning models. It transforms genomic sequences into predicted chromatin accessibility profiles and quantifies the "disruption" caused by sequence alterations.

The toolkit moves from raw variant lists to summarized, annotated effect sizes and contribution (SHAP) scores.

## Repository Layout

```text
src/
  variant_scoring.py           # Core script: predicts scores for variant alleles
  variant_scoring.per_chrom.py # Memory-efficient scoring (per-chromosome)
  variant_summary_across_folds.py # Aggregates results from multi-fold models
  variant_annotation.py        # Overlaps variants with peaks, genes, or motifs
  variant_shap.py              # Computes base-resolution contribution scores
  
output/                        # All script outputs (TSV, HDF5) should be directed here
  {prefix}.variant_scores.tsv  # Standard output from scoring
  {prefix}.variant_shap.h5     # SHAP contribution scores

```

## Data Specifications

### Variant Input Schemas

Variants must be provided as TSVs. The `--schema` argument defines the expected columns:

| Schema | Columns |
| --- | --- |
| `chrombpnet` | `['chr', 'pos', 'allele1', 'allele2', 'variant_id']` |
| `bed` | `['chr', 'pos', 'end', 'allele1', 'allele2', 'variant_id']` |
| `plink` | `['chr', 'variant_id', 'ignore1', 'pos', 'allele1', 'allele2']` |
| `original` | `['chr', 'pos', 'variant_id', 'allele1', 'allele2']` |

> **Note on Indexing:** The `pos` column is **1-indexed** for all schemas except `bed`, which follows the standard 0-indexed BED convention.

### Allele Representations

* **SNPs:** `A`, `C`, `G`, `T` in both allele columns.
* **Deletions:** Use `-` for `allele2`.
* **Insertions:** Use `-` for `allele1`.

## Key Metrics & Definitions

Effect sizes are computed as `allele2` (variant) vs `allele1` (reference).

* **`logfc`**: Log2 fold-change of total predicted coverage. Positive values indicate the variant increases accessibility.
* **`jsd`**: Jensen-Shannon Distance. Measures how much the *shape* of the accessibility profile changes (e.g., a TF footprint disappearing), independent of total magnitude.
* **`active_allele_quantile` (AAQ)**: Percentile of the stronger allele's coverage relative to all peaks in the training data.
* **Integrative Scores**:
* **IES (Integrative Effect Size)**: `abs_logfc * jsd`.
* **IPS (Integrative Prioritization Score)**: `logfc * jsd * AAQ`.


## Workflow & Execution

### 1. Primary Scoring

Use `variant_scoring.py` for standard runs. For large whole-genome lists, use `variant_scoring.per_chrom.py` to avoid OOM (Out of Memory) errors.


### 2. Multi-Fold Aggregation

ChromBPNet models are often trained in cross-validation folds. We average over the folds to improve robustness.


### 3. Interpretation (SHAP)

To see *why* a variant is scored highly, compute SHAP scores. This reveals which nucleotides (the variant itself or flanking motifs) drive the prediction.

* **Outputs:** HDF5 file containing `raw/seq` (one-hot) and `projected_shap/seq` (importance values).

## Technical Constraints

* **Forward/Reverse Bias:** By default, predictions are averaged across the forward and reverse-complement sequences. Use `--forward_only` only if specifically required by the experimental design.
* **Logits vs Probs:** Profile predictions are saved as **logits**. When averaging across folds, average the logits first, then apply softmax.
* **Memory Management:** For large variant sets, always use `--no_hdf5` unless base-pair resolution bigwigs are strictly required for every variant.
