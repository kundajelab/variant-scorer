import os
import h5py
import argparse
import numpy as np
import pandas as pd

def trim_ppm(ppm, t=0.45, min_length=3, flank=0):
    # trim matrix to first and last bp that have
    # p>=threshold 
    maxes = np.max(ppm,-1)
    maxes = np.where(maxes>=t)

    # if no bases with prob>t or too small:
    if (len(maxes[0])==0) or (maxes[0][-1]+1-maxes[0][0]<min_length):
        return None
    
    return ppm[max(maxes[0][0]-flank, 0):maxes[0][-1]+1+flank]

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



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--modisco_h5py", required=True, type=str)
    parser.add_argument("-mt", "--modisco_tomtom", required=True, type=str)
    parser.add_argument("-o", "--output_file", required=True, type=str)
    parser.add_argument("-t", "--threshold", default=0.45, type=float)
    parser.add_argument("-ml", "--min_length", default=3, type=int, help="Min length of acceptable motif")
    parser.add_argument("-f", "--flank_add", default=0, type=int, help="Add this on either side of motif after trimming")
    parser.add_argument("-s", "--min_seqlets", default=100, type=int, help="Minimum seqlets associated with motif")
    parser.add_argument("-n", "--normalize", default=False, action='store_true', help="PPM (probability) instead of PFM (frequency)" )
    args = parser.parse_args()
    
    tomtom_results = pd.read_csv(args.modisco_tomtom, sep="\t")
    tomtom_results["best_matches"] = tomtom_results["Match_1"] + "_" + tomtom_results.index.astype(str)
    f = h5py.File(args.modisco_h5py, 'r')
    output_file = open(args.output_file, "a")
    write_prelim_lines(output_file)
    num_patterns = len(f['metacluster_idx_to_submetacluster_results']['metacluster_0']['seqlets_to_patterns_result']['patterns'])-1
    
    for i in range(num_patterns):
        trimmed_ppm = trim_ppm(f['metacluster_idx_to_submetacluster_results']['metacluster_0']['seqlets_to_patterns_result']['patterns']['pattern_{}'.format(i)]['sequence']['fwd'], 
                t=args.threshold, min_length=args.min_length, flank=args.flank_add)

        if trimmed_ppm is not None:
            num_seqlets = len(f['metacluster_idx_to_submetacluster_results']['metacluster_0']['seqlets_to_patterns_result']['patterns']['pattern_{}'.format(i)]['seqlets_and_alnmts']['seqlets'])

            if num_seqlets >= args.min_seqlets:
                pfm = (trimmed_ppm*num_seqlets).astype(int)
            
                if args.normalize:
                    pfm = (pfm+1)/np.sum(pfm, axis=1, keepdims=True)
                    pfm = np.where(pfm > 1, 1, pfm)
                curr_motif = tomtom_results.loc[i, "best_matches"]
                curr_motif = "NoMatch" if (type(curr_motif) == float and np.isnan(curr_motif)) else curr_motif
                write_for_one_motif(pfm, curr_motif, output_file)
    
    f.close()
    
