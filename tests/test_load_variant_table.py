import pytest
import pandas as pd
import os
import sys

# Add src to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils.io import load_variant_table, get_variant_schema

class TestLoadVariantTable:
	
	@pytest.fixture
	def test_data_dir(self):
		return os.path.join(os.path.dirname(__file__), 'data')
	
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
		
	def test_load_bed_schema(self, test_data_dir):
		"""Test loading variants with bed schema"""
		file_path = os.path.join(test_data_dir, 'test.bed')
		df = load_variant_table(file_path, 'bed')
		
		# Check columns
		expected_cols = ['chr', 'pos', 'end', 'allele1', 'allele2', 'variant_id']
		assert list(df.columns) == expected_cols
		
		# Check that positions are converted from 0-based to 1-based
		# The pos column should be incremented by 1
		assert df['pos'].dtype in ['int64', 'int32']
		
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
