import pytest
import subprocess
import tempfile
import os
import sys
import importlib.util
import numpy as np
import pandas as pd
import h5py
from pathlib import Path

class TestVariantScoringCLI:

	@pytest.fixture(scope="class")
	def script_path(self, src):
		"""Fixture to provide the path to the variant_scoring.py script"""
		return os.path.join(src, "variant_scoring.py")

	@pytest.fixture(scope="class")
	def script_path_per_chrom(self, src):
		"""Fixture to provide the path to the variant_scoring.per_chrom.py script"""
		return os.path.join(src, "variant_scoring.per_chrom.py")

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
		"""Test variant scoring accuracy against known caQTLs (depends on summary test)"""
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

	@pytest.mark.gpu
	@pytest.mark.oak
	def test_variant_scoring_per_chrom(self, out_dir, script_path_per_chrom, test_data_dir, genome_path, model_paths, chrom_sizes_path):
		"""Test variant_scoring.per_chrom.py with real genome/model data (requires env vars and GPU)"""
		if not os.path.exists(script_path_per_chrom):
			pytest.skip(f"Script {script_path_per_chrom} not found")
		
		test_variants = os.path.join(test_data_dir, 'test.chrombpnet.tsv')
		if not os.path.exists(test_variants):
			pytest.skip("Test variant file not found")

		# Check inputs
		input_df = pd.read_csv(test_variants, sep='\t', header=None)
		chrms = input_df[0].unique()

		# Dictionary of number of variants per chromosome
		variant_counts = input_df[0].value_counts().to_dict()

		# Run for fold 0
		model_path = model_paths[0]
		output_prefix = os.path.join(out_dir, f"fold_0")

		cmd = [
			sys.executable, script_path_per_chrom,
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
		
		# Check output files exist
		df_list = []
		for chr in chrms:
			output_file = f"{output_prefix}.{chr}.variant_scores.tsv"
			assert os.path.exists(output_file), f"Output file for {chr} not created"

			df = pd.read_csv(output_file, sep='\t')
			assert 'logfc' in df.columns, "Missing logfc column"
			assert 'jsd' in df.columns, "Missing jsd column"

			# Check that we have the right number of variants
			expected_count = variant_counts.get(chr, 0)
			assert len(df) == expected_count, f"Output for {chr} has {len(df)} variants, expected {expected_count}"

			df_list.append(df)

	@pytest.mark.gpu
	@pytest.mark.oak
	def test_merge_chroms(self, out_dir, script_path_per_chrom, test_data_dir, genome_path, model_paths, chrom_sizes_path):
		"""Test variant_scoring.per_chrom.py merges scores/predictions successfully with real genome/model data (requires env vars and GPU)"""
		if not os.path.exists(script_path_per_chrom):
			pytest.skip(f"Script {script_path_per_chrom} not found")
		
		test_variants = os.path.join(test_data_dir, 'test.chrombpnet.tsv')
		if not os.path.exists(test_variants):
			pytest.skip("Test variant file not found")

		# Check inputs
		input_df = pd.read_csv(test_variants, sep='\t', header=None)
		chrms = input_df[0].unique()

		# Run for fold 0
		model_path = model_paths[0]
		output_prefix = os.path.join(out_dir, f"with_merge_fold_0")

		cmd = [
			sys.executable, script_path_per_chrom,
			'--list', test_variants,
			'--genome', genome_path,
			'--model', model_path,
			'--out_prefix', output_prefix,
			'--chrom_sizes', chrom_sizes_path,
			'--num_shuf', '2',  # Use a small number for testing
			'--schema', 'chrombpnet',
			"--merge"
		]
			
		result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
	
		if result.returncode != 0:
			print(f"STDOUT: {result.stdout}")
			print(f"STDERR: {result.stderr}")
	
		# Check if it completed successfully
		assert result.returncode == 0, f"Script failed: {result.stderr}"

		# Load variant scores
		output_file = f"{output_prefix}.variant_scores.tsv"
		df = pd.read_csv(output_file, sep='\t')

		assert 'logfc' in df.columns, "Missing logfc column"
		assert 'jsd' in df.columns, "Missing jsd column"

		# Check number of variants
		input_df = pd.read_csv(test_variants, sep='\t', header=None)
		assert len(df) == len(input_df), "Merged output has different number of variants than input"

		# Load predictions & check h5 format and shape
		output_h5 = f"{output_prefix}.variant_predictions.h5"
		with h5py.File(output_h5, 'r') as f:
			
			assert "observed" in f, "No 'observed' group in predictions file"
			observed = f["observed"]
			assert "allele1_pred_counts" in observed, "No 'allele1_pred_counts' dataset in predictions file"
			assert "allele2_pred_counts" in observed, "No 'allele2_pred_counts' dataset in predictions file"
			assert "allele1_pred_profiles" in observed, "No 'allele1_pred_profiles' dataset in predictions file"
			assert "allele2_pred_profiles" in observed, "No 'allele2_pred_profiles' dataset in predictions file"

			assert observed["allele1_pred_counts"].shape[0] == len(df), f"Predictions have {observed['allele1_pred_counts'].shape[0]} variants, expected {len(df)}"
			assert observed["allele1_pred_profiles"].shape[0] == len(df), f"Predictions have {observed['allele1_pred_profiles'].shape[0]} variants, expected {len(df)}"


class TestMergeH5Predictions:
	"""Unit tests for merge_h5_predictions() — no GPU or Oak data required."""

	@pytest.fixture(scope="class")
	def merge_fn(self, src):
		"""Import merge_h5_predictions from utils.merge."""
		spec = importlib.util.spec_from_file_location(
			"merge",
			os.path.join(src, "utils", "merge.py")
		)
		mod = importlib.util.module_from_spec(spec)
		spec.loader.exec_module(mod)
		return mod.merge_h5_predictions

	def _write_chrom_h5(self, path, n_variants, profile_len=100):
		"""Write a synthetic per-chromosome H5 in the expected format."""
		with h5py.File(path, 'w') as f:
			obs = f.create_group('observed')
			obs.create_dataset('allele1_pred_counts',  data=np.random.rand(n_variants))
			obs.create_dataset('allele2_pred_counts',  data=np.random.rand(n_variants))
			obs.create_dataset('allele1_pred_profiles', data=np.random.rand(n_variants, profile_len))
			obs.create_dataset('allele2_pred_profiles', data=np.random.rand(n_variants, profile_len))

	def test_merge_combines_chroms(self, merge_fn, tmp_path):
		"""Merged file contains all four datasets concatenated across chromosomes."""
		counts = [5, 7, 3]
		chrom_files = []
		for i, n in enumerate(counts):
			p = str(tmp_path / f"out.chr{i+1}.variant_predictions.h5")
			self._write_chrom_h5(p, n)
			chrom_files.append(p)

		merged = str(tmp_path / "out.variant_predictions.h5")
		merge_fn(chrom_files, merged)

		total = sum(counts)
		with h5py.File(merged, 'r') as f:
			assert 'observed' in f
			obs = f['observed']
			for key in ['allele1_pred_counts', 'allele2_pred_counts',
						'allele1_pred_profiles', 'allele2_pred_profiles']:
				assert key in obs, f"Missing dataset: {key}"
			assert obs['allele1_pred_counts'].shape[0]   == total
			assert obs['allele2_pred_counts'].shape[0]   == total
			assert obs['allele1_pred_profiles'].shape[0] == total
			assert obs['allele2_pred_profiles'].shape[0] == total

	def test_merge_deletes_chrom_files(self, merge_fn, tmp_path):
		"""Per-chromosome H5 files are deleted after merging."""
		counts = [4, 6]
		chrom_files = []
		for i, n in enumerate(counts):
			p = str(tmp_path / f"del.chr{i+1}.variant_predictions.h5")
			self._write_chrom_h5(p, n)
			chrom_files.append(p)

		merged = str(tmp_path / "del.variant_predictions.h5")
		merge_fn(chrom_files, merged)

		for f in chrom_files:
			assert not os.path.exists(f), f"Expected {f} to be deleted after merge"

	def test_merge_missing_file_skipped(self, merge_fn, tmp_path):
		"""A missing per-chrom file is skipped gracefully and the rest are merged."""
		p_exists = str(tmp_path / "skip.chr1.variant_predictions.h5")
		p_missing = str(tmp_path / "skip.chr2.variant_predictions.h5")
		self._write_chrom_h5(p_exists, 5)

		merged = str(tmp_path / "skip.variant_predictions.h5")
		merge_fn([p_exists, p_missing], merged)

		with h5py.File(merged, 'r') as f:
			assert f['observed']['allele1_pred_counts'].shape[0] == 5