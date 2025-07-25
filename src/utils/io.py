import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon


def get_variant_schema(schema):
    var_SCHEMA = {'original': ['chr', 'pos', 'variant_id', 'allele1', 'allele2'],
                  'plink': ['chr', 'variant_id', 'ignore1', 'pos', 'allele1', 'allele2'],
                  'plink2': ['chr', 'variant_id', 'pos', 'allele1', 'allele2'],
                  'bed': ['chr', 'pos', 'end', 'allele1', 'allele2', 'variant_id'],
                  'chrombpnet': ['chr', 'pos', 'allele1', 'allele2', 'variant_id']}
    return var_SCHEMA[schema]


def get_peak_schema(schema):
    PEAK_SCHEMA = {'narrowpeak': ['chr', 'start', 'end', 'peak_id', 'peak_score',
                                  5, 6, 7, 'rank', 'summit']}
    return PEAK_SCHEMA[schema]


def validate_alleles(variants_table):
    """Validate that alleles contain only valid nucleotides (ACGT) or deletion marker (-)"""
    valid_chars = set('ACGT-')
    
    for col in ['allele1', 'allele2']:
        if col in variants_table.columns:
            for idx, allele in enumerate(variants_table[col]):

                allele_str = str(allele).upper()

                if not set(allele_str).issubset(valid_chars):
                    raise ValueError(f"Invalid characters in {col} at row {idx}: '{allele}'. Only A, C, G, T, and - are allowed.")

                # If the allele contains "-", it should be a single character
                if '-' in allele_str and len(allele_str) > 1:
                    raise ValueError(f"Invalid allele at row {idx}: '{allele}'. Use a single '-' to represent INDELs.")


def load_variant_table(table_path, schema):
    # Read file first to check structure
    temp_df = pd.read_csv(table_path, header=None, sep='\t', nrows=5)
    expected_cols = len(get_variant_schema(schema))
    
    if temp_df.shape[1] != expected_cols:
        raise ValueError(f"File has {temp_df.shape[1]} columns but {schema} schema expects {expected_cols} columns")
    
    variants_table = pd.read_csv(table_path, header=None, sep='\t', names=get_variant_schema(schema))
    variants_table.drop(columns=[str(x) for x in variants_table.columns if str(x).startswith('ignore')], inplace=True)
    variants_table['chr'] = variants_table['chr'].astype(str)
    has_chr_prefix = any('chr' in x.lower() for x in variants_table['chr'].tolist())
    if not has_chr_prefix:
        variants_table['chr'] = 'chr' + variants_table['chr']
    if schema == "bed":
        # Convert to 1-based indexing
        variants_table['pos'] = variants_table['pos'] + 1
    
    # Validate alleles
    validate_alleles(variants_table)
    
    return variants_table


def add_missing_columns_to_peaks_df(peaks, schema):
    if schema != 'narrowpeak':
        raise ValueError("Schema not supported")
    
    required_columns = get_peak_schema(schema)
    num_current_columns = peaks.shape[1]
    
    if num_current_columns == 10:
        peaks.columns = required_columns[:num_current_columns]
        return peaks  # No missing columns, return as is

    elif num_current_columns < 3:
        raise ValueError("Peaks dataframe has fewer than 3 columns, which is invalid")
    
    elif num_current_columns > 10:
        raise ValueError("Peaks dataframe has greater than 10 columns, which is invalid")
    
    # Add missing columns to reach a total of 10 columns
    peaks.columns = required_columns[:num_current_columns]
    columns_to_add = required_columns[num_current_columns:]
    
    for column in columns_to_add:
        peaks[column] = '.'
    
    # Calculate the summit column
    peaks['summit'] = (peaks['end'] - peaks['start']) // 2
    
    return peaks
