from tensorflow.keras.utils import Sequence
import pandas as pd
import numpy as np
import math
import pyfaidx
from utils import one_hot
from deeplift.dinuc_shuffle import dinuc_shuffle

class PeakGenerator(Sequence):
    def __init__(self,
                 peaks,
                 input_len,
                 genome_fasta,
                 batch_size=512,
                 debug_mode=False):

        self.peaks = peaks
        self.num_peaks = self.peaks.shape[0]
        self.input_len = input_len
        self.genome = pyfaidx.Fasta(genome_fasta)
        self.debug_mode = debug_mode
        self.flank_size = self.input_len // 2

    def __get_seq__(self, chrom, start, summit):
        chrom = str(chrom)
        start = int(start)
        summit = int(start) + int(summit)
        flank_start = int(summit - self.flank_size)
        flank_end = int(summit + (self.flank_size - 1))
        flank = str(self.genome.get_seq(chrom, flank_start, flank_end))
        return flank

    def __getitem__(self, idx):
            
        cur_entries = self.peaks.iloc[idx*self.batch_size:min([self.num_peaks,(idx+1)*self.batch_size])]
        seqs = zip(*[self.__get_seq__(x, y, z) for x,y,z in
                     zip(cur_entries.chr, cur_entries.start, cur_entries.summit)])
        

        return one_hot.dna_to_one_hot(list(seqs))

    def __len__(self):
        return math.ceil(self.num_peaks/self.batch_size)

    
