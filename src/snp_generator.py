from tensorflow.keras.utils import Sequence
import pandas as pd
import numpy as np
import math
import pyfaidx
from utils import one_hot
from deeplift.dinuc_shuffle import dinuc_shuffle

class SNPGenerator(Sequence):
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
        if self.shuf:
            self.batch_size = 32
        else:
            self.batch_size = batch_size

    def __get_allele_seq__(self, chrom, pos, allele1, allele2, seed=0):
        chrom = str(chrom)
        pos = int(pos)
        allele1 = str(allele1)
        allele2 = str(allele2)
        flank_start = int(pos - self.flank_size)
        flank_end = int(pos + (self.flank_size - 1))
        flank = str(self.genome.get_seq(chrom, flank_start, flank_end))

        if self.shuf:
            flank = dinuc_shuffle(flank, rng=np.random.RandomState(seed))

        allele1_seq = flank[:self.flank_size] + allele1 + flank[self.flank_size+1:]
        allele2_seq = flank[:self.flank_size] + allele2 + flank[self.flank_size+1:]
        return allele1_seq, allele2_seq

    def __getitem__(self, idx):
        cur_entries = self.variants_table.iloc[idx*self.batch_size:min([self.num_variants,(idx+1)*self.batch_size])]
        rsids = cur_entries['rsid'].tolist()

        if self.shuf:
            rsids = np.repeat(rsids, 10)
            allele1_seqs, allele2_seqs = zip(*[self.__get_allele_seq__(v, w, x, y, z) for v,w,x,y,z in
                                             zip(np.repeat(cur_entries.chr, 10), np.repeat(cur_entries.pos, 10),
                                                 np.repeat(cur_entries.allele1, 10), np.repeat(cur_entries.allele2, 10),
                                                 np.tile([1234, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999], len(cur_entries)))])
        else:
            allele1_seqs, allele2_seqs = zip(*[self.__get_allele_seq__(w, x, y, z) for w,x,y,z in
                                             zip(cur_entries.chr, cur_entries.pos, cur_entries.allele1, cur_entries.allele2)])

        return rsids, one_hot.dna_to_one_hot(list(allele1_seqs)), one_hot.dna_to_one_hot(list(allele2_seqs))

    def __len__(self):
        return math.ceil(self.num_variants/self.batch_size)

