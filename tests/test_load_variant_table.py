import pytest
import pandas as pd
import os
import sys

# Add src to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils.io import load_variant_table, get_variant_schema

class TestLoadVariantTable:
	
	def test_load_chrombpnet_schema(self, test_data_dir):
		"""Test loading variants with chrombpnet schema"""
		file_path = os.path.join(test_data_dir, 'test.chrombpnet.tsv')
		df = load_variant_table(file_path, 'chrombpnet')
		
		# Check columns
		expected_cols = ['chr', 'pos', 'allele1', 'allele2', 'variant_id']
		assert list(df.columns) == expected_cols
		
		# Check data types
		assert df['chr'].dtype == 'object'
		assert df['pos'].dtype in ['int64', 'int32']
		
		# Check chromosome prefixes are preserved/added
		assert all(df['chr'].str.startswith('chr'))
		
		# Check basic structure
		assert len(df) > 0
		assert df['allele1'].notna().all()
		assert df['allele2'].notna().all()

	def test_incorrect_chrombpnet(self, test_data_dir):
		"""Test loading an incorrect chrombpnet file"""
		file_path = os.path.join(test_data_dir, 'test.chrombpnet.incorrect.tsv')
		
		with pytest.raises(ValueError):
			load_variant_table(file_path, 'chrombpnet')
		
	def test_load_bed_schema(self, test_data_dir):
		"""Test loading variants with bed schema"""
		file_path = os.path.join(test_data_dir, 'test.bed')
		
		# First check that the file has the right number of columns
		df_orig = pd.read_csv(file_path, sep='\t', header=None)
		expected_bed_cols = 6
		assert df_orig.shape[1] == expected_bed_cols, f"BED file should have {expected_bed_cols} columns, found {df_orig.shape[1]}"
		
		df = load_variant_table(file_path, 'bed')
		
		# Check columns
		expected_cols = ['chr', 'pos', 'end', 'allele1', 'allele2', 'variant_id']
		assert list(df.columns) == expected_cols
		
		# Check that no columns have NaN values
		assert not df.isnull().any().any(), "BED file has missing values"
		
		# Check that the position column is incremented by 1
		assert (df['pos'] - 1).equals(df_orig[1])

	def test_incorrect_bed(self, test_data_dir):
		"""Test loading an incorrect bed file"""
		file_path = os.path.join(test_data_dir, 'test.incorrect.bed')
		
		with pytest.raises(ValueError):
			load_variant_table(file_path, 'bed')

	def test_load_plink_schema(self, test_data_dir):
		"""Test loading variants with plink schema"""
		file_path = os.path.join(test_data_dir, 'test.plink.tsv')
		df = load_variant_table(file_path, 'plink')
		
		# Check remaining columns
		expected_cols = ['chr', 'variant_id', 'pos', 'allele1', 'allele2']
		assert list(df.columns) == expected_cols
		
	def test_chromosome_prefix_addition(self, test_data_dir):
		"""Test that chr prefix is added when missing"""
		# Look for files without chr prefix

		file_path = os.path.join(test_data_dir, 'test.chrombpnet.no_chr.tsv')
		df = load_variant_table(file_path, 'chrombpnet')
			
		# Check that chr prefix was added
		assert all(df['chr'].str.startswith('chr'))
	
	def test_invalid_alleles(self, test_data_dir):
		"""Test that files with invalid allele characters are rejected"""
		file_path = os.path.join(test_data_dir, 'test.chrombpnet.incorrect.tsv')
		
		with pytest.raises(ValueError, match="Invalid characters"):
			load_variant_table(file_path, 'chrombpnet')

	def test_invalid_alleles2(self, test_data_dir):
		"""Test that files with invalid allele characters are rejected"""
		file_path = os.path.join(test_data_dir, 'test.chrombpnet.incorrect2.tsv')
		
		with pytest.raises(ValueError, match="Invalid allele"):
			load_variant_table(file_path, 'chrombpnet')
	
	def test_invalid_schema(self, test_data_dir):
		"""Test that invalid schema raises appropriate error"""
		file_path = os.path.join(test_data_dir, 'test.chrombpnet.tsv')
		
		with pytest.raises(KeyError):
			load_variant_table(file_path, 'invalid_schema')
	
	def test_file_not_found(self):
		"""Test that missing file raises appropriate error"""
		with pytest.raises(FileNotFoundError):
			load_variant_table('nonexistent_file.tsv', 'chrombpnet')
	
	def test_get_variant_schema(self):
		"""Test the get_variant_schema helper function"""
		# Test all known schemas
		schemas = ['original', 'plink', 'bed', 'chrombpnet']
		
		for schema in schemas:
			columns = get_variant_schema(schema)
			assert isinstance(columns, list)
			assert len(columns) > 0
			assert 'chr' in columns
			assert 'pos' in columns
		
		# Test invalid schema
		with pytest.raises(KeyError):
			get_variant_schema('invalid_schema')
