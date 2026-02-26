import torch
import tangermeme.predict
from bpnetlite import BPNet
from scipy.spatial.distance import jensenshannon
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('..')
from generators.variant_generator import VariantGenerator
from generators.peak_generator import PeakGenerator
from utils.wrappers import ChromBPNetWrapper
from utils.io import get_variant_schema, get_peak_schema


def get_valid_peaks(chrom, pos, summit, input_len, chrom_sizes_dict):
	valid_chrom = chrom in chrom_sizes_dict
	if valid_chrom:
		flank = input_len // 2
		lower_check = ((pos + summit) - flank > 0)
		upper_check = ((pos + summit) + flank <= chrom_sizes_dict[chrom])
		in_bounds = lower_check and upper_check
		valid_peak = valid_chrom and in_bounds
		return valid_peak
	else:
		return False

def get_valid_variants(chrom, pos, allele1, allele2, input_len, chrom_sizes_dict):
	valid_chrom = chrom in chrom_sizes_dict
	if valid_chrom:
		flank = input_len // 2
		lower_check = (pos - flank > 0)
		upper_check = (pos + flank <= chrom_sizes_dict[chrom])
		in_bounds = lower_check and upper_check
		valid_variant = valid_chrom and in_bounds
		return valid_variant
	else:
		return False

def softmax(x, temp=1):
	norm_x = x - np.mean(x, axis=1, keepdims=True)
	return np.exp(temp*norm_x)/np.sum(np.exp(temp*norm_x), axis=1, keepdims=True)

def load_bpnet_model(model_file):
	"""Load a ChromBPNet .h5 model file and wrap it with ChromBPNetWrapper.

	Uses bpnetlite's BPNet.from_chrombpnet() to convert the TF .h5 file to
	a PyTorch model, then wraps it for convenient profile + counts output.

	Sets model.input_len and model.bpnet for downstream access.
	"""
	bpnet = BPNet.from_chrombpnet(filename=model_file)
	model = ChromBPNetWrapper(bpnet)
	model.eval()
	# For ChromBPNet models, output_len is always 1000 bp;
	# bpnet.trimming = (input_len - output_len) // 2, so input_len = 1000 + 2*trimming
	model.input_len = 1000 + 2 * bpnet.trimming
	model.bpnet = bpnet
	print("model loaded successfully")
	return model

def fetch_peak_predictions(model, peaks, input_len, genome_fasta, batch_size, debug_mode=False, lite=False, forward_only=False):
	peak_ids = []
	pred_counts = []
	pred_profiles = []
	if not forward_only:
		revcomp_counts = []
		revcomp_profiles = []

	# peak sequence generator
	peak_gen = PeakGenerator(peaks=peaks,
							 input_len=input_len,
							 genome_fasta=genome_fasta,
							 batch_size=batch_size,
							 debug_mode=debug_mode)

	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	for i in tqdm(range(len(peak_gen))):
		batch_peak_ids, seqs = peak_gen[i]
		# seqs shape: (N, 4, L) channels-first
		revcomp_seq = seqs[:, ::-1, ::-1].copy()

		seqs_tensor = torch.tensor(seqs, dtype=torch.float32)
		revcomp_tensor = torch.tensor(revcomp_seq, dtype=torch.float32)

		batch_preds = tangermeme.predict.predict(model, seqs_tensor, device=device)
		# batch_preds: [profile (N, 1, L) or (N, L), log_counts (N, 1) or (N,)]
		batch_counts = np.exp(batch_preds[1].squeeze(-1).cpu().numpy())
		batch_profile = batch_preds[0].squeeze(1).cpu().numpy()

		pred_counts.extend(batch_counts)
		pred_profiles.extend(batch_profile)

		if not forward_only:
			revcomp_preds = tangermeme.predict.predict(model, revcomp_tensor, device=device)
			revcomp_c = np.exp(revcomp_preds[1].squeeze(-1).cpu().numpy())
			revcomp_p = revcomp_preds[0].squeeze(1).cpu().numpy()
			revcomp_counts.extend(revcomp_c)
			revcomp_profiles.extend(revcomp_p)

		peak_ids.extend(batch_peak_ids)

	peak_ids = np.array(peak_ids)
	pred_counts = np.array(pred_counts)
	pred_profiles = np.array(pred_profiles)

	if not forward_only:
		revcomp_counts = np.array(revcomp_counts)
		revcomp_profiles = np.array(revcomp_profiles)
		average_counts = np.average([pred_counts, revcomp_counts], axis=0)
		average_profiles = np.average([pred_profiles, revcomp_profiles[:, ::-1]], axis=0)
		return peak_ids, average_counts, average_profiles
	else:
		return peak_ids, pred_counts, pred_profiles

def fetch_variant_predictions(model, variants_table, input_len, genome_fasta, batch_size, debug_mode=False, lite=False, shuf=False, forward_only=False):
	variant_ids = []
	allele1_pred_counts = []
	allele2_pred_counts = []
	allele1_pred_profiles = []
	allele2_pred_profiles = []
	if not forward_only:
		revcomp_allele1_pred_counts = []
		revcomp_allele2_pred_counts = []
		revcomp_allele1_pred_profiles = []
		revcomp_allele2_pred_profiles = []

	# variant sequence generator
	var_gen = VariantGenerator(variants_table=variants_table,
						   input_len=input_len,
						   genome_fasta=genome_fasta,
						   batch_size=batch_size,
						   debug_mode=False,
						   shuf=shuf)

	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	for i in tqdm(range(len(var_gen))):

		batch_variant_ids, allele1_seqs, allele2_seqs = var_gen[i]
		# seqs shape: (N, 4, L) channels-first
		revcomp_allele1_seqs = allele1_seqs[:, ::-1, ::-1].copy()
		revcomp_allele2_seqs = allele2_seqs[:, ::-1, ::-1].copy()

		allele1_tensor = torch.tensor(allele1_seqs, dtype=torch.float32)
		allele2_tensor = torch.tensor(allele2_seqs, dtype=torch.float32)

		a1_preds = tangermeme.predict.predict(model, allele1_tensor, device=device)
		a2_preds = tangermeme.predict.predict(model, allele2_tensor, device=device)

		a1_counts = np.exp(a1_preds[1].squeeze(-1).cpu().numpy())
		a2_counts = np.exp(a2_preds[1].squeeze(-1).cpu().numpy())
		a1_profiles = a1_preds[0].squeeze(1).cpu().numpy()
		a2_profiles = a2_preds[0].squeeze(1).cpu().numpy()

		allele1_pred_counts.extend(a1_counts)
		allele2_pred_counts.extend(a2_counts)
		allele1_pred_profiles.extend(a1_profiles)
		allele2_pred_profiles.extend(a2_profiles)

		if not forward_only:
			rc_a1_tensor = torch.tensor(revcomp_allele1_seqs, dtype=torch.float32)
			rc_a2_tensor = torch.tensor(revcomp_allele2_seqs, dtype=torch.float32)

			rc_a1_preds = tangermeme.predict.predict(model, rc_a1_tensor, device=device)
			rc_a2_preds = tangermeme.predict.predict(model, rc_a2_tensor, device=device)

			revcomp_allele1_pred_counts.extend(np.exp(rc_a1_preds[1].squeeze(-1).cpu().numpy()))
			revcomp_allele2_pred_counts.extend(np.exp(rc_a2_preds[1].squeeze(-1).cpu().numpy()))
			revcomp_allele1_pred_profiles.extend(rc_a1_preds[0].squeeze(1).cpu().numpy())
			revcomp_allele2_pred_profiles.extend(rc_a2_preds[0].squeeze(1).cpu().numpy())

		variant_ids.extend(batch_variant_ids)

	variant_ids = np.array(variant_ids)
	allele1_pred_counts = np.array(allele1_pred_counts)
	allele2_pred_counts = np.array(allele2_pred_counts)
	allele1_pred_profiles = np.array(allele1_pred_profiles)
	allele2_pred_profiles = np.array(allele2_pred_profiles)

	if not forward_only:
		revcomp_allele1_pred_counts = np.array(revcomp_allele1_pred_counts)
		revcomp_allele2_pred_counts = np.array(revcomp_allele2_pred_counts)
		revcomp_allele1_pred_profiles = np.array(revcomp_allele1_pred_profiles)
		revcomp_allele2_pred_profiles = np.array(revcomp_allele2_pred_profiles)
		average_allele1_pred_counts = np.average([allele1_pred_counts, revcomp_allele1_pred_counts], axis=0)
		average_allele1_pred_profiles = np.average([allele1_pred_profiles, revcomp_allele1_pred_profiles[:, ::-1]], axis=0)
		average_allele2_pred_counts = np.average([allele2_pred_counts, revcomp_allele2_pred_counts], axis=0)
		average_allele2_pred_profiles = np.average([allele2_pred_profiles, revcomp_allele2_pred_profiles[:, ::-1]], axis=0)
		return variant_ids, average_allele1_pred_counts, average_allele2_pred_counts, \
			   average_allele1_pred_profiles, average_allele2_pred_profiles
	else:
		return variant_ids, allele1_pred_counts, allele2_pred_counts, \
			   allele1_pred_profiles, allele2_pred_profiles

def get_variant_scores_with_peaks(allele1_pred_counts, allele2_pred_counts,
				   allele1_pred_profiles, allele2_pred_profiles, pred_counts):
	logfc, jsd = get_variant_scores(allele1_pred_counts, allele2_pred_counts,
									allele1_pred_profiles, allele2_pred_profiles)
	allele1_quantile = np.array([np.max([np.mean(pred_counts < x), (1/len(pred_counts))]) for x in allele1_pred_counts])
	allele2_quantile = np.array([np.max([np.mean(pred_counts < x), (1/len(pred_counts))]) for x in allele2_pred_counts])

	return logfc, jsd, allele1_quantile, allele2_quantile

def get_variant_scores(allele1_pred_counts, allele2_pred_counts,
				   allele1_pred_profiles, allele2_pred_profiles):

	print('allele1_pred_counts shape:', allele1_pred_counts.shape)
	print('allele2_pred_counts shape:', allele2_pred_counts.shape)
	print('allele1_pred_profiles shape:', allele1_pred_profiles.shape)
	print('allele2_pred_profiles shape:', allele2_pred_profiles.shape)

	logfc = np.squeeze(np.log2(allele2_pred_counts / allele1_pred_counts))
	jsd = np.squeeze([jensenshannon(x, y, base=2.0)
					 for x,y in zip(softmax(allele2_pred_profiles),
									softmax(allele1_pred_profiles))])

	print('logfc shape:', logfc.shape)
	print('jsd shape:', jsd.shape)

	return logfc, jsd

def adjust_indel_jsd(variants_table,allele1_pred_profiles,allele2_pred_profiles,original_jsd):
	allele1_pred_profiles = softmax(allele1_pred_profiles)
	allele2_pred_profiles = softmax(allele2_pred_profiles)
	indel_idx = []
	for i, row in variants_table.iterrows():
		allele1, allele2 = row[['allele1','allele2']]
		if allele1 == "-":
			allele1 = ""
		if allele2 == "-":
			allele2 = ""
		if len(allele1) != len(allele2):
			indel_idx += [i]

	adjusted_jsd = []
	for i in indel_idx:
		row = variants_table.iloc[i]
		allele1, allele2 = row[['allele1','allele2']]
		if allele1 == "-":
			allele1 = ""
		if allele2 == "-":
			allele2 = ""

		allele1_length = len(allele1)
		allele2_length = len(allele2)

		allele1_p = allele1_pred_profiles[i]
		allele2_p = allele2_pred_profiles[i]
		assert len(allele1_p) == len(allele2_p)
		assert allele1_length != allele2_length
		flank_size = len(allele1_p)//2
		allele1_left_flank = allele1_p[:flank_size]
		allele2_left_flank = allele2_p[:flank_size]

		if allele1_length > allele2_length:
			allele1_right_flank = np.concatenate([allele1_p[flank_size:flank_size+allele2_length],allele1_p[flank_size+allele1_length:]])
			allele2_right_flank = allele2_p[flank_size:allele2_length-allele1_length]
		else:
			allele1_right_flank = allele1_p[flank_size:allele1_length-allele2_length]
			allele2_right_flank = np.concatenate([allele2_p[flank_size:flank_size+allele1_length], allele2_p[flank_size+allele2_length:]])


		adjusted_allele1_p = np.concatenate([allele1_left_flank,allele1_right_flank])
		adjusted_allele2_p = np.concatenate([allele2_left_flank,allele2_right_flank])
		adjusted_allele1_p = adjusted_allele1_p/np.sum(adjusted_allele1_p)
		adjusted_allele2_p = adjusted_allele2_p/np.sum(adjusted_allele2_p)
		assert len(adjusted_allele1_p) == len(adjusted_allele2_p)
		adjusted_j = jensenshannon(adjusted_allele1_p,adjusted_allele2_p,base=2.0)
		adjusted_jsd += [adjusted_j]

	adjusted_jsd_list = original_jsd.copy()
	if len(indel_idx) > 0:
		for i in range(len(indel_idx)):
			idx = indel_idx[i]
			adjusted_jsd_list[idx] = adjusted_jsd[i]

	return indel_idx, adjusted_jsd_list


def create_shuffle_table(variants_table, random_seed=None, total_shuf=None, num_shuf=None):
	if total_shuf != None:
		if len(variants_table) > total_shuf:
			shuf_variants_table = variants_table.sample(total_shuf,
														random_state=random_seed,
														ignore_index=True,
														replace=False)
		else:
			shuf_variants_table = variants_table.sample(total_shuf,
														random_state=random_seed,
														ignore_index=True,
														replace=True)
		shuf_variants_table['random_seed'] = np.random.permutation(len(shuf_variants_table))
	else:
		if num_shuf != None:
			total_shuf = len(variants_table) * num_shuf
			shuf_variants_table = variants_table.sample(total_shuf,
														random_state=random_seed,
														ignore_index=True,
														replace=True)
			shuf_variants_table['random_seed'] = np.random.permutation(len(shuf_variants_table))
		else:
			## empty dataframe
			shuf_variants_table = pd.DataFrame()
	return shuf_variants_table

def get_pvals(obs, bg, tail):
	sorted_bg = np.sort(bg)
	if tail == 'right' or tail == 'both':
		rank_right = len(sorted_bg) - np.searchsorted(sorted_bg, obs, side='left')
		pval_right = (rank_right + 1) / (len(sorted_bg) + 1)
		if tail == 'right':
			return pval_right
	if tail == 'left' or tail == 'both':
		rank_left = np.searchsorted(sorted_bg, obs, side='right')
		pval_left = (rank_left + 1) / (len(sorted_bg) + 1)
		if tail == 'left':
			return pval_left
	assert tail == 'both'
	min_pval = np.minimum(pval_left, pval_right)
	pval_both = min_pval * 2

	return pval_both
