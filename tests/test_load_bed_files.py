import pytest
import pandas as pd
import pybedtools
import os
import sys

# Add src to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils.io import load_genes, load_peaks

class TestLoadBedFiles:

	def test_load_genes(self, test_data_dir):
		"""Test loading genes file"""
		file_path = os.path.join(test_data_dir, 'test.genes.bed')
		gene_bed = load_genes(file_path)
		
		# Check that it returns a BedTool object
		assert isinstance(gene_bed, pybedtools.BedTool)
		
		# Check that it has data
		assert len(gene_bed) > 0
		
		# Check basic BED format (at least 3 columns)
		df = gene_bed.to_dataframe()
		assert df.shape[1] >= 3
	
	def test_load_peaks(self, test_data_dir):
		"""Test loading peaks file"""
		file_path = os.path.join(test_data_dir, 'test.peaks.bed')
		peak_bed = load_peaks(file_path)
		
		# Check that it returns a BedTool object
		assert isinstance(peak_bed, pybedtools.BedTool)
		
		# Check that it has data
		assert len(peak_bed) > 0
		
		# Check basic BED format (at least 3 columns)
		df = peak_bed.to_dataframe()
		assert df.shape[1] >= 3

	
	def test_file_not_found_genes(self):
		"""Test that missing genes file raises appropriate error"""
		with pytest.raises(FileNotFoundError):
			load_genes('nonexistent_genes.bed')
	
	def test_file_not_found_peaks(self):
		"""Test that missing peaks file raises appropriate error"""
		with pytest.raises(FileNotFoundError):
			load_peaks('nonexistent_peaks.bed')
	