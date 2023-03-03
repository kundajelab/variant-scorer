from tensorflow.keras.utils import Sequence
import pandas as pd
import numpy as np
import math
import pyfaidx
from utils import one_hot
from deeplift.dinuc_shuffle import dinuc_shuffle

class VariantGenerator(Sequence):
    def __init__(self,
                 variants_table,
                 input_len,
                 genome_fasta,
                 batch_size=512,
                 debug_mode=False,
                 shuf=False):

        self.variants_table = variants_table
        self.num_variants = self.variants_table.shape[0]
        self.input_len = input_len
        self.genome = pyfaidx.Fasta(genome_fasta)
        self.debug_mode = debug_mode
        self.flank_size = self.input_len // 2
        self.shuf = shuf
        self.batch_size = batch_size

    def __get_allele_seq__(self, chrom, pos, allele1, allele2, seed=-1):
        chrom = str(chrom)
        pos = int(pos)
        allele1 = str(allele1)
        allele2 = str(allele2)
        
        if allele1 == "-":
            allele1 = ""
        if allele2 == "-":
            allele2 = ""
        ### 1 - indexed position 
        pos = pos - 1
        
        assert self.genome[chrom][pos:pos+len(allele1)] == allele1
        if len(allele1) == 1 and len(allele2) == 1:
            flank  = str(self.genome[chrom][pos-self.flank_size:pos+self.flank_size])
            if self.shuf:
                assert seed != -1
                flank = dinuc_shuffle(flank, rng=np.random.RandomState(seed))
            allele1_seq = flank[:self.flank_size] + allele1 + flank[self.flank_size+1:]
            allele2_seq = flank[:self.flank_size] + allele2 + flank[self.flank_size+1:]

        ### handle INDELS (allele1 must be the reference allele)
        else:
            mismatch_length = len(allele1) - len(allele2)
            if mismatch_length > 0:
                flank = str(self.genome[chrom][pos-self.flank_size:pos+self.flank_size+mismatch_length])
            else:
                flank = str(self.genome[chrom][pos-self.flank_size:pos+self.flank_size])

            if self.shuf:
                assert seed != -1
                flank = dinuc_shuffle(flank, rng=np.random.RandomState(seed))

            left_flank=flank[:self.flank_size]

            allele1_right_flank = flank[self.flank_size+len(allele1):self.flank_size*2]
            allele2_right_flank = flank[self.flank_size+len(allele1):self.flank_size*2+mismatch_length]
            
            allele1_seq = left_flank + allele1 + allele1_right_flank
            allele2_seq = left_flank + allele2 + allele2_right_flank

        assert len(allele1_seq) == self.flank_size * 2
        assert len(allele2_seq) == self.flank_size * 2
        return allele1_seq, allele2_seq

    def __getitem__(self, idx):
        cur_entries = self.variants_table.iloc[idx*self.batch_size:min([self.num_variants,(idx+1)*self.batch_size])]
        rsids = cur_entries['rsid'].tolist()

        if self.shuf:
            allele1_seqs, allele2_seqs = zip(*[self.__get_allele_seq__(v, w, x, y, z) for v,w,x,y,z in
                                             zip(cur_entries.chr, cur_entries.pos,
                                                 cur_entries.allele1, cur_entries.allele2, cur_entries.random_seed)])
        else:
            allele1_seqs, allele2_seqs = zip(*[self.__get_allele_seq__(w, x, y, z) for w,x,y,z in
                                             zip(cur_entries.chr, cur_entries.pos, cur_entries.allele1, cur_entries.allele2)])

        if self.debug_mode:
            return rsids, list(allele1_seqs),list(allele2_seqs)
        else:
            return rsids, one_hot.dna_to_one_hot(list(allele1_seqs)), one_hot.dna_to_one_hot(list(allele2_seqs))
        

    def __len__(self):
        return math.ceil(self.num_variants/self.batch_size)

