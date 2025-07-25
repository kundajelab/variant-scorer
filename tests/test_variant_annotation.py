import pytest
import subprocess
import os
import sys
import pandas as pd
from pathlib import Path

class TestVariantAnnotationCLI:

	@pytest.fixture(scope="class")
	def script_path(self, src):
		"""Fixture to provide the path to the variant_annotation.py script"""
		return os.path.join(src, "variant_annotation.py")


	def test_arg_validation(self, test_data_dir, script_path, out_dir):
		"""Test that an error is raised if at least one of genes, peaks, or hits is not provided."""

		test_variants = os.path.join(test_data_dir, 'test.bed')
		if not os.path.exists(test_variants):
			pytest.skip("Test variant file not found")

		output_prefix = os.path.join(out_dir, f"scores")

		cmd = [
			sys.executable, script_path,
			'--list', test_variants,
			'--schema', 'bed',
			'--out_prefix', output_prefix
		]

		# Run the command and check for error
		result = subprocess.run(cmd, capture_output=True, text=True)

		assert result.returncode != 0, "Command should fail without peaks, hits, or genes"

		# Check stdout instead of stderr since that's where the error message appears
		assert "at least one of" in result.stdout.lower(), \
			f"Expected error message not found in stdout. Got: {result.stdout}"


	def test_variant_annotation(self, test_data_dir, script_path, out_dir):
		"""Test the variant annotation script with valid inputs."""
		
		test_variants = os.path.join(test_data_dir, 'test.anno_input.tsv')
		if not os.path.exists(test_variants):
			pytest.skip("Test variant file not found")

		output_prefix = os.path.join(out_dir, f"scores")

		cmd = [
			sys.executable, script_path,
			'--list', test_variants,
			'--out_prefix', output_prefix,
			'--peaks', os.path.join(test_data_dir, 'test.peaks.bed'),
			'--hits', os.path.join(test_data_dir, 'test.hits.bed'),
			'--genes', os.path.join(test_data_dir, 'test.genes.bed')
		]

		result = subprocess.run(cmd, capture_output=True, text=True)

		assert result.returncode == 0, f"Command failed with error: {result.stderr}"

		output_file = f"{output_prefix}.annotations.tsv"
		assert os.path.exists(output_file), "Output file was not created"

		df = pd.read_csv(output_file, sep='\t')
		
		assert not df.empty, "Output DataFrame is empty"

		# Test that the hit overlaps are correct:
		expected_motifs = ['motif_aa', '-', 'motif_cc', 'motif_dd', '-', '-', '-', 'motif_gg,motif_cc', '-', 'motif_hh']
		expected_motifs2 = ['motif_aa', '-', 'motif_cc', 'motif_dd', '-', '-', '-', 'motif_cc,motif_gg', '-', 'motif_hh']
		assert df['hits_motifs'].tolist() == expected_motifs or df['hits_motifs'].tolist() == expected_motifs2, \
			f"Expected hits motifs {expected_motifs} but got {df['hits_motifs'].tolist()}"

		expected_peak_overlap = [True, False, False, False, False, True, False, True, False, True]
		assert df['peak_overlap'].tolist() == expected_peak_overlap, \
			f"Expected peak overlap {expected_peak_overlap} but got {df['peak_overlap'].tolist()}"

		expected_closest_genes = ['gene_B', 'gene_C', 'gene_C', 'gene_C', 'gene_C', 'gene_D', 'gene_G', 'gene_G', 'gene_H', 'gene_H']
		assert df['closest_gene_1'].tolist() == expected_closest_genes, \
			f"Expected closest genes {expected_closest_genes} but got {df['closest_gene_1'].tolist()}"
