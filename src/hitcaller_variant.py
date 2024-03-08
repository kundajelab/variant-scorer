import numpy as np 
import h5py
import subprocess
import argparse
import os
import pandas as pd


def parse_args():
	parser = argparse.ArgumentParser(description="Runs hit calling on variant-based interpretation data")
	parser.add_argument("--shap_data", type=str, required=True, help="h5 or npz file containing variant sequences and shap_scores")
	parser.add_argument("--input_type", type=str, choices=["h5", "npz"], default="h5", help="Whether the input data is in h5 or npz format")
	parser.add_argument("--modisco_h5", type=str, help="Modisco h5 file from relevant experiment")
	parser.add_argument("--variant_file", type=str, help="variant-scorer style file containing list of variants. Required if you want genomic locations as part of the final report")
	parser.add_argument("--hits_per_loc", type=int, help="Maximum number of hits to return per sequence per locus")
	parser.add_argument("--output_dir", type=str, help="Output directory")
	parser.add_argument("--alpha", type=float, default=0.6, help="Alpha value for hit calling")
	args = parser.parse_args()
	return args


def h5_to_npz(args):
	'''
	If the input is given as a h5, then this function runs the relevant finemo command to convert it to a npz file
	Regions of width 100 are extracted, since this should be sufficient for hits containing the central variant
	'''
	extract_command = ["finemo", "extract-regions-h5", "-c", args.shap_data, "-w", "100", "-o", os.path.join(args.output_dir, "shap_input.npz")]
	subprocess.run(extract_command)

def run_hit_calling(args, npz_file):
	'''
	Runs hit calling given the npz file with input interpretation data
	'''
	if args.variant_file is not None:
		subprocess.run(["finemo", "call-hits", "-r", npz_file, "-m", args.modisco_h5, "-o", args.output_dir, "-b", "1000", "-a", str(args.alpha), "-p", os.path.join(args.output_dir, "variant_locs.narrowPeak")])
	else:
		subprocess.run(["finemo", "call-hits", "-r", npz_file, "-m", args.modisco_h5, "-o", args.output_dir, "-b", "1000", "-a", str(args.alpha)])

def parse_hit_calls(args):
	'''
	Given an output file from the hit caller, identifies hits containing the central variant and returns the top n hits per sequence
	'''
	hits_file = os.path.join(args.output_dir, "hits_unique.tsv")
	hits_df = pd.read_csv(hits_file, sep="\t")
	# print(hits_df.head())

	#Define location of variants to identify correct hits
	if args.variant_file is not None:
		variant_table = pd.read_csv(args.variant_file, sep="\t", header=None)
		hits_df["variant_loc"] = variant_table.loc[(hits_df["peak_id"] % len(variant_table)).astype(int), 1].values
		print(hits_df.head())
	else:
		hits_df["variant_loc"] = [50] * len(hits_df)

	variant_hits = hits_df.loc[(hits_df["start"] <= hits_df["variant_loc"]) & (hits_df["end"] >= hits_df["variant_loc"])]
	variant_hits["inv_coeff"] = -1 * variant_hits["hit_coefficient"]
	print()
	print(variant_hits.head())
	variant_hits = variant_hits.sort_values(["peak_id", "inv_coeff"]).groupby("peak_id").head(args.hits_per_loc)
	if args.variant_file is not None:
		variant_hits['allele'] = variant_hits['peak_id'].apply(lambda x: "allele2" if x > len(variant_table) else "allele1")
	else:
		variant_hits['allele'] - "N/A"
	variant_out_final = variant_hits[["peak_id", "chr", "start", "end", "motif_name", "allele",
								   	  "variant_loc", "hit_coefficient", "hit_correlation", "hit_importance"]]
	return variant_out_final


def variant_file_to_narrowpeak(args):
	'''
	Converts a variant info file (ie. the input to most variant-scorer commands) into a narrowpeak file which can be used with the hit caller
	'''
	variant_table = pd.read_csv(args.variant_file, sep="\t", header=None)
	narrowpeak_raw_data = [list(variant_table[0].values), list(variant_table[1].values - 1), list(variant_table[1].values + 1),
                      ["."] * len(variant_table), ["."] * len(variant_table), ["."] * len(variant_table), ["."] * len(variant_table),
                      ["."] * len(variant_table), ["."] * len(variant_table), [1] * len(variant_table)]
	narrowpeak_df = pd.DataFrame(narrowpeak_raw_data).T
	narrowpeak_df = pd.concat([narrowpeak_df, narrowpeak_df])

	narrowpeak_df.to_csv(os.path.join(args.output_dir, "variant_locs.narrowPeak"), sep="\t", header=False, index=False)
	return narrowpeak_df


def main():

	#Produce npz file if it does not already exist
	args = parse_args()
	if args.input_type == "npz":
		npz_file = args.shap_data
	elif args.input_type == "h5":
		h5_to_npz(args)
		npz_file = os.path.join(args.output_dir, "shap_input.npz")

	#Produce narrowpeak file if desired
	if args.variant_file is not None:
		npeak = variant_file_to_narrowpeak(args)

	#Run the hit caller and save the results
	run_hit_calling(args, npz_file)
	output_df = parse_hit_calls(args)
	output_df.to_csv(args.output_dir + "variant_hit_calls.tsv", sep="\t", header=True, index=False)


if __name__ == "__main__":
	main()



