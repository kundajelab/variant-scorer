import pytest
import subprocess
import tempfile
import os
import sys
import pandas as pd
from pathlib import Path

class TestVariantScoringCLI:

	@pytest.fixture(scope="class")
	def script_path(self, src):
		"""Fixture to provide the path to the variant_scoring.py script"""
		return os.path.join(src, "variant_scoring.py")

	@pytest.mark.oak
	def test_variant_scoring_help(self, script_path):
		"""Test that variant_scoring.py shows help without errors"""
		if not os.path.exists(script_path):
			pytest.skip(f"Script {script_path} not found")
		
		cmd = [sys.executable, script_path, '--help']
		result = subprocess.run(cmd, capture_output=True, text=True)
		
		# Should exit successfully and show help
		assert result.returncode == 0
		assert 'usage:' in result.stdout.lower() or 'help' in result.stdout.lower()
		# Check that required arguments are mentioned
		assert '--list' in result.stdout
		assert '--genome' in result.stdout
		assert '--model' in result.stdout
		assert '--out_prefix' in result.stdout
		assert '--chrom_sizes' in result.stdout
	
	@pytest.mark.oak
	def test_variant_scoring_missing_required_args(self, script_path):
		"""Test that variant_scoring.py fails gracefully with missing required arguments"""
		if not os.path.exists(script_path):
			pytest.skip(f"Script {script_path} not found")
		
		cmd = [sys.executable, script_path]
		result = subprocess.run(cmd, capture_output=True, text=True)
		
		# Should fail with non-zero exit code
		assert result.returncode != 0

		# Should mention missing required arguments
		error_text = result.stderr.lower()
		assert 'required' in error_text or 'argument' in error_text or 'missing' in error_text

	@pytest.mark.gpu
	@pytest.mark.oak
	def test_variant_scoring_no_peaks(self, out_dir, script_path, test_data_dir, genome_path, model_paths, chrom_sizes_path):
		"""Test variant_scoring.py with real genome/model data (requires env vars and GPU)"""
		if not os.path.exists(script_path):
			pytest.skip(f"Script {script_path} not found")
		
		test_variants = os.path.join(test_data_dir, 'test.chrombpnet.tsv')
		if not os.path.exists(test_variants):
			pytest.skip("Test variant file not found")

		# Run for each fold
		for fold in range(5):
			model_path = model_paths[fold]
			output_prefix = os.path.join(out_dir, f"fold_{fold}")

			cmd = [
				sys.executable, script_path,
				'--list', test_variants,
				'--genome', genome_path,
				'--model', model_path,
				'--out_prefix', output_prefix,
				'--chrom_sizes', chrom_sizes_path,
				'--num_shuf', '2',  # Use a small number for testing
				'--schema', 'chrombpnet',
				'--no_hdf5'  # Skip HDF5 output for faster testing
			]
				
			result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
		
			if result.returncode != 0:
				print(f"STDOUT: {result.stdout}")
				print(f"STDERR: {result.stderr}")
		
			# Check if it completed successfully
			assert result.returncode == 0, f"Script failed: {result.stderr}"
			
			# Check output file exists
			output_file = f"{output_prefix}.variant_scores.tsv"
			assert os.path.exists(output_file), "Output file not created"
			
			# Basic validation of output
			df = pd.read_csv(output_file, sep='\t')
			assert len(df) > 0, "Output file is empty"
			assert 'logfc' in df.columns, "Missing logfc column"
			assert 'jsd' in df.columns, "Missing jsd column"
			
			# Validate that we have the expected number of variants
			input_df = pd.read_csv(test_variants, sep='\t', header=None)
			assert len(df) == len(input_df), "Output has different number of variants than input"

	@pytest.mark.oak
	def test_variant_summary_across_folds(self, out_dir, script_path, test_data_dir):
		"""Test variant_summary_across_folds.py (depends on scoring test)"""
		# This test depends on the scoring test having run successfully
		# Check that fold output files exist
		for fold in range(5):
			output_file = os.path.join(out_dir, f"fold_{fold}.variant_scores.tsv")
			if not os.path.exists(output_file):
				pytest.skip("Scoring test outputs not found. Requires test_variant_scoring_no_peaks.")

		# Run summary script
		summary_script = os.path.join(os.path.dirname(script_path), 'variant_summary_across_folds.py')
		if not os.path.exists(summary_script):
			pytest.skip(f"Summary script {summary_script} not found")

		summary_cmd = [
			sys.executable, summary_script,
			'--score_dir', out_dir,
			'--score_list'
		]
		# Add each file as a separate argument
		for fold in range(5):
			summary_cmd.append(f"fold_{fold}.variant_scores.tsv")
		
		summary_cmd.extend([
			'--out_prefix', os.path.join(out_dir, 'summary'),
			'--schema', 'chrombpnet'
		])

		result = subprocess.run(summary_cmd, capture_output=True, text=True)
		assert result.returncode == 0, f"Summary script failed: {result.stderr}"

		# Check output file exists
		summary_file = os.path.join(out_dir, 'summary.mean.variant_scores.tsv')
		assert os.path.exists(summary_file), "Summary output file not created"

	@pytest.mark.oak
	def test_variant_scoring_accuracy(self, out_dir, test_data_dir):
		"""Test variant scoring accuracy against known CaQTLs (depends on summary test)"""
		# Check that summary output exists
		summary_file = os.path.join(out_dir, 'summary.mean.variant_scores.tsv')
		if not os.path.exists(summary_file):
			pytest.skip("Summary test output not found. Requires test_variant_summary_across_folds.")
		
		# Load CaQTL reference data
		caqtl_file = os.path.join(test_data_dir, 'caqtls.african.lcls.benchmarking.subset.tsv')
		if not os.path.exists(caqtl_file):
			pytest.skip("CaQTL reference file not found")
		
		# Load scoring results
		scores_df = pd.read_csv(summary_file, sep='\t')
		caqtl_df = pd.read_csv(caqtl_file, sep='\t')
		
		# Basic validation
		assert len(scores_df) > 0, "No scoring results found"
		assert len(caqtl_df) > 0, "No CaQTL data found"
		
		# Check that we have the expected columns
		assert 'logfc.mean' in scores_df.columns, "Missing logfc.mean column in scores"
		assert 'pred.chrombpnet.encsr000emt.variantscore.logfc' in caqtl_df.columns, "Missing logfc column in ground truth"
		
		# Merge datasets on variant identifier
		if 'variant_id' in scores_df.columns and 'var.dbsnp_rsid' in caqtl_df.columns:
			merged_df = pd.merge(scores_df, caqtl_df, left_on='variant_id', right_on='var.dbsnp_rsid', how='inner')
			assert len(merged_df) == len(scores_df), "No overlapping variants found between scores and CaQTL data"

			# Check tolerance between predicted and ground truth logfc
			tolerance = 1e-3 # Adjust as needed
			diff = abs(merged_df['logfc.mean'] - merged_df['pred.chrombpnet.encsr000emt.variantscore.logfc'])
			
			# This will always show in pytest output
			print(f"\nDifference stats:")
			print(diff)
			
			within_tolerance = (diff <= tolerance)

			# check that all the variants are within tolerance
			assert all(within_tolerance), f"Not all variants within tolerance: {within_tolerance.sum()}/{len(merged_df)}"

		else:
			pytest.skip("Cannot merge datasets - missing variant_id or dbsnp_rsid column")


