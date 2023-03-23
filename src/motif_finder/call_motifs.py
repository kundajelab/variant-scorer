import numpy as np
import os
import subprocess
import deepdish
import tempfile
import sys
sys.path.append("../../ancient_human_regulation/")
from utils.viz_sequence import *

def write_meme_file(ppm, bg, fname):
	f = open(fname, 'w')
	f.write('MEME version 4\n\n')
	f.write('ALPHABET= ACGT\n\n')
	f.write('strands: + -\n\n')
	f.write('Background letter frequencies (from unknown source):\n')
	f.write('A %.3f C %.3f G %.3f T %.3f\n\n' % tuple(list(bg)))
	f.write('MOTIF 1 TEMP\n')
	f.write('letter-probability matrix: alength= 4 w= %d nsites= 200\n' % ppm.shape[0])
	for s in ppm:
		f.write('%.5f %.5f %.5f %.5f\n' % tuple(s))
	f.write('URL none\n\n')
	f.close()

def fetch_tomtom_matches(ppm, cwm, background=[0.25, 0.25, 0.25, 0.25], tomtom_exec_path='/oak/stanford/groups/akundaje/patelas/software/meme_4.11.2/tomtom', motifs_db='/oak/stanford/groups/akundaje/soumyak/motifs/motifs.meme.txt', n=5, trim_threshold=0.3, trim_min_length=3):

	"""Fetches top matches from a motifs database using TomTom.
	Args:
		ppm: position probability matrix- numpy matrix of dimension (N,4)
		background: list with ACGT background probabilities
		tomtom_exec_path: path to TomTom executable
		motifs_db: path to motifs database in meme format
		n: number of top matches to return, ordered by p-value
		temp_dir: directory for storing temp files
		trim_threshold: the ppm is trimmed from left till first position for which
			probability for any base pair >= trim_threshold. Similarly from right.
	Returns:
		list: a list of up to n results returned by tomtom, each entry is a
			dictionary with keys 'Target ID', 'p-value', 'E-value', 'q-value'
	"""

	_, fname = tempfile.mkstemp()

	score = np.sum(np.abs(cwm), axis=1)
	trim_thresh = np.max(score) * 0.25  # Cut off anything less than 30% of max score
	pass_inds = np.where(score >= trim_thresh)[0]
	trimmed = ppm[np.min(pass_inds): np.max(pass_inds) + 1]

	# can be None of no base has prob>t
	if trimmed is None:
		return []

	# trim and prepare meme file
	write_meme_file(trimmed, background, fname)

	# run tomtom
	cmd = '%s -no-ssc -oc . -verbosity 1 -text -min-overlap 5 -mi 1 -dist pearson -evalue -thresh 10.0 %s %s' % (tomtom_exec_path, fname, motifs_db)
	#print(cmd)
	out = subprocess.check_output(cmd, shell=True)
	# prepare output
	dat = [x.split('\\t') for x in str(out).split('\\n')]
	schema = dat[0]

	# meme v4 vs v5:
	if 'Target ID' in schema:
		tget_idx = schema.index('Target ID')
	else:
		tget_idx = schema.index('Target_ID')

	pval_idx, eval_idx, qval_idx =schema.index('p-value'), schema.index('E-value'), schema.index('q-value')

	r = []
	for t in dat[1:min(1+n, len(dat)-1)]:
		if t[0]=='':
			break

		mtf = {}
		mtf['Target_ID'] = t[tget_idx]
		mtf['p-value'] = float(t[pval_idx])
		mtf['E-value'] = float(t[eval_idx])
		mtf['q-value'] = float(t[qval_idx])
		r.append(mtf)

	os.system('rm ' + fname)
	return r

def get_center_and_rest(shap_heights, center_length):
	'''
	Given a 1-D array of SHAP scores, divides this array into center and flanks
	The center will be the array of length center_length centered at the midpoint
	The "rest" will be the rest of the array concatenated together.
	'''
	center_seq = shap_heights[len(shap_heights) // 2 - center_length // 2 : len(shap_heights) // 2 + center_length // 2]
	rest_seq = np.concatenate([shap_heights[:len(shap_heights) // 2 - center_length // 2], shap_heights[len(shap_heights) // 2 + center_length // 2 :]])
	return center_seq, rest_seq

def get_quantile(shap_heights, quantile):
	'''
	Returns the n% quantile of a given set of shap_scores
	'''
	return np.quantile(shap_heights, quantile)

def locate_motif(shap_heights, threshold, count_to_exit):
	'''
	Given a 1-D array of importance scores, determines the approximate location of a motif if it exists.
	Assumes the motif is at the center of the array. 
	Inputs are:
	`shap_heights:  1-D importance score array
	`threshold: importance score threshold to consider a position part of the motif
	`count_to_exit: number of positions below the threshold before we decide to exit the motif
	Basically, this works by starting at the center, and then working our way to the left and right
	We add all positions above the threshold to the motif
	We exit on one side when we encounter count_to_exit consecutive positions below the threshold
	We finish when we exit on both sides
	Returns the starting and ending position with respect to the center
	'''
	#Start both left and right at 0
	left_start, right_start = len(shap_heights) // 2, len(shap_heights) // 2
	#Currently, we are still trying both sides
	left_running, right_running = True, True
	#We haven't encountered any below the threshold
	left_small, right_small = 0, 0
	while left_running or right_running:
		#If we are still going on the left, do the following
		if left_running:
			#If higher than the threshold, we keep going
			#We also reset the left_small counter
			if shap_heights[left_start] > threshold:
				left_start -= 1
				left_small = 0
			else:
				#Otherwise, we add one to left_small
				#If that equals count_to_exit, then we are done on the left side
				left_small += 1
				left_start -= 1
				if left_small == count_to_exit:
					left_running = False
					#We add one because we just subtracted one
					#Left_start is the final starting position
					left_start += count_to_exit + 1
		if right_running:
			#We do the same thing for the right side
			if shap_heights[right_start] > threshold:
				right_start += 1
				right_small = 0
			else:
				right_small += 1
				right_start += 1
				if right_small == count_to_exit:
					right_running = False
					right_start -= (count_to_exit + 1)
	return min(left_start - len(shap_heights) // 2, 0),  max(right_start - len(shap_heights) // 2, 0)
	
def find_motif_loc(shap_scores, center_length=30, quantile=0.95, count_to_exit=2):
	'''
	Finds the motif location in a set of SHAP scores
	Inputs are 
	`shap_scores: 4xN array representing a set of importance scores
	`center_length: length of center of array to search for motif in
	`quantile: quantile of non-center array that motif must be higher than
	`count_to_exit: the number of positions below the quantile before we decide to exit the motif
	'''
	#Convert shap scores to 1-D array
	position_heights = shap_scores.sum(0)
	#Define where we look for the motif, and where we define the background
	center, rest = get_center_and_rest(position_heights, center_length)
	#Use quantile to get threshold
	threshold = get_quantile(rest, quantile)
	#Get motif bounds
	left_bound, right_bound = locate_motif(center, threshold, count_to_exit)
	return len(position_heights) // 2 + left_bound, len(position_heights) // 2 + right_bound

def get_match(shap_scores, motifs_db):
	quants = [0.99, 0.98, 0.95]
	counts = [2,1]
	for q in quants:
		for c in counts:
			try:
				start, end = find_motif_loc(shap_scores, quantile=q, count_to_exit=c)
				cwm = shap_scores[:,start:end + 1].T
				ppm = cwm != 0
				matches = fetch_tomtom_matches(ppm, cwm, motifs_db=motifs_db)
				if len(matches) > 0:
					return matches, matches[0]["Target_ID"]
			except:
				continue
	return "None", "None"
