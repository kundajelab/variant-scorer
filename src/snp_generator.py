from tensorflow.keras.utils import Sequence
import pandas as pd
import numpy as np
import math
import pyfaidx
from utils import one_hot

class SNPGenerator(Sequence):
    def __init__(self,
                 variants_table,
                 input_len,
                 genome_fasta,
                 batch_size=128,
                 debug_mode=False):

        self.variants_table = variants_table
        self.num_variants = self.variants_table.shape[0]
        self.input_len = input_len
        self.batch_size = batch_size
        self.genome = pyfaidx.Fasta(genome_fasta)
        self.debug_mode = debug_mode

    def __getitem__(self,idx):
        
        allele1_seqs = []
        allele2_seqs = []
            
        cur_entries = self.variants_table.iloc[idx*self.batch_size:min([self.num_variants,(idx+1)*self.batch_size])]
        flank_size = self.input_len // 2

        rsids = cur_entries['rsid'].tolist()

        for index,entry in cur_entries.iterrows():

            chrom = str(entry["chr"])
            pos = int(entry["pos"])
            rsid = str(entry["rsid"])
            allele1 = str(entry["allele1"])
            allele2 = str(entry["allele2"])

            flank_start = int(pos - flank_size)
            flank_end = int(pos + (flank_size - 1))
            flank = str(self.genome.get_seq(chrom, flank_start, flank_end))
            assert len(flank) == self.input_len
            cur_allele1_seq = str(flank)[:flank_size] + allele1 + str(flank)[flank_size+1:]
            cur_allele2_seq = str(flank)[:flank_size] + allele2 + str(flank)[flank_size+1:]

            #if self.debug_mode:
            #    print(rsid)
            #    print("allele1:")
            #    print(cur_allele1_seq[flank_size-5:flank_size+5])
            #    print("allele2:")
            #    print(cur_allele2_seq[flank_size-5:flank_size+5])
           
            allele1_seqs.append(cur_allele1_seq)
            allele2_seqs.append(cur_allele2_seq)

        return rsids, one_hot.dna_to_one_hot(allele1_seqs), one_hot.dna_to_one_hot(allele2_seqs)

    def __len__(self):
        return math.ceil(self.num_variants/self.batch_size)
