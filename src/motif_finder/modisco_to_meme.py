import os
import h5py
import argparse
import numpy as np
import pandas as pd

# def trim_ppm(ppm, t=0.45, min_length=3, flank=0):
#     # trim matrix to first and last bp that have
#     # p>=threshold
#     maxes = np.max(ppm,-1)
#     maxes = np.where(maxes>=t)

#     # if no bases with prob>t or too small:
#     if (len(maxes[0])==0) or (maxes[0][-1]+1-maxes[0][0]<min_length):
#         return None

#     return ppm[max(maxes[0][0]-flank, 0):maxes[0][-1]+1+flank]

def write_prelim_lines(out_file):
    out_file.write("MEME version 4\n")
    out_file.write("\n")
    out_file.write("ALPHABET= ACGT\n")
    out_file.write("\n")
    out_file.write("strands: + -\n")
    out_file.write("\n")
    out_file.write("Background letter frequencies\n")
    out_file.write("A 0.25 C 0.25 G 0.25 T 0.25\n")
    out_file.write("\n")

def write_for_one_motif(pfm, motif, out_file):
    out_file.write("MOTIF " + motif + "\n")
    out_file.write("letter-probability matrix: alength= 4 w= %i nsites= 200\n"%(len(pfm)))
    for row in pfm:
        out_file.write(' '.join([str(x) for x in row]) + "\n")
    out_file.write("URL none\n")
    out_file.write("\n")

def trim_motif(ppm, cwm, background=[0.25, 0.25, 0.25, 0.25],trim_threshold=0.3):
    score = np.sum(np.abs(cwm), axis=1)
    trim_thresh = np.max(score) * trim_threshold  # Cut off anything less than 30% of max score
    pass_inds = np.where(score >= trim_thresh)[0]
    trimmed = ppm[np.min(pass_inds): np.max(pass_inds) + 1]

    # can be None of no base has prob>t
    if trimmed is None:
        trimmed = []
    return trimmed

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--modisco_h5py", required=True, type=str)
    parser.add_argument("-mt", "--modisco_tomtom", required=True, type=str)
    parser.add_argument("-o", "--output_prefix", required=True, type=str)
    parser.add_argument("-t", "--threshold", default=0.3, type=float)
    parser.add_argument("-ml", "--min_length", default=3, type=int, help="Min length of acceptable motif")
    parser.add_argument("-f", "--flank_add", default=0, type=int, help="Add this on either side of motif after trimming")
    parser.add_argument("-s", "--min_seqlets", default=20, type=int, help="Minimum seqlets associated with motif")
    parser.add_argument("-n", "--normalize", default=True, action='store_true', help="PPM (probability) instead of PFM (frequency)" )
    args = parser.parse_args()

    tomtom_results = pd.read_csv(args.modisco_tomtom, sep="\t")
    tomtom_results.set_index('pattern', inplace=True)
    tomtom_results["best_matches"] = tomtom_results["match0"] + "--" + tomtom_results.index.astype(str)
    modisco_results = h5py.File(args.modisco_h5py, 'r')

    for name in ['pos_patterns', 'neg_patterns']:
        if name not in modisco_results.keys():
            continue

        if name == 'pos_patterns':
            output_file = open(args.output_prefix + '.pos.meme.txt', "w")
        else:
            output_file = open(args.output_prefix + '.neg.meme.txt', "w")

        write_prelim_lines(output_file)
        metacluster = modisco_results[name]
        key = lambda x: int(x[0].split("_")[-1])
        for pattern_name, pattern in sorted(metacluster.items(), key=key):
            ppm = np.array(pattern['sequence'][:])
            cwm = np.array(pattern["contrib_scores"][:])

            trimmed = trim_motif(ppm, cwm,trim_threshold=args.threshold)
            if len(trimmed) > args.min_length:
                num_seqlets = pattern['seqlets']['n_seqlets'][0]
                if num_seqlets >= args.min_seqlets:
                    curr_motif = tomtom_results.at[name + "." + pattern_name, "best_matches"]
                    curr_motif = "NoMatch" if (type(curr_motif) == float and np.isnan(curr_motif)) else curr_motif
                    write_for_one_motif(trimmed, curr_motif, output_file)

    modisco_results.close()

