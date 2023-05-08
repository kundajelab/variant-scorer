from tensorflow.keras.utils import Sequence
import pandas as pd
import numpy as np
import math
import pyfaidx
from utils import one_hot
from deeplift.dinuc_shuffle import dinuc_shuffle
import warnings

class VariantGenerator(Sequence):
    def __init__(
        self,
        variants_table,
        input_len,
        genome_fasta,
        batch_size=512,
        debug_mode=False,
        shuf=False,
        return_coords=False,
    ):
        self.variants_table = variants_table
        self.num_variants = self.variants_table.shape[0]
        self.input_len = input_len
        self.genome = pyfaidx.Fasta(genome_fasta)
        self.debug_mode = debug_mode
        self.flank_size = self.input_len // 2
        self.shuf = shuf
        self.batch_size = batch_size
        self.return_coords = return_coords

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

        if len(allele1) == len(allele2):
            flank = str(
                self.genome[chrom][pos - self.flank_size : pos + self.flank_size]
            )
            if self.shuf:
                assert seed != -1
                flank = dinuc_shuffle(flank, rng=np.random.RandomState(seed))
            allele1_seq = (
                flank[: self.flank_size]
                + allele1
                + flank[self.flank_size + len(allele1) :]
            )
            allele2_seq = (
                flank[: self.flank_size]
                + allele2
                + flank[self.flank_size + len(allele2) :]
            )

        ### handle INDELS (allele1 must be the reference allele)
        else:
            ### hg19 has lower case
            assert len(allele1) != len(allele2)
            assert self.genome[chrom][pos : pos + len(allele1)].seq.upper() == allele1
            mismatch_length = len(allele1) - len(allele2)
            if mismatch_length > 0:  # deletion
                flank = str(
                    self.genome[chrom][
                        pos - self.flank_size : pos + self.flank_size + mismatch_length
                    ]
                )
            else:  # insertion
                flank = str(
                    self.genome[chrom][pos - self.flank_size : pos + self.flank_size]
                )

            if self.shuf:
                assert seed != -1
                flank = dinuc_shuffle(flank, rng=np.random.RandomState(seed))

            left_flank = flank[: self.flank_size]

            allele1_right_flank = flank[
                self.flank_size + len(allele1) : self.flank_size * 2
            ]
            allele2_right_flank = flank[
                self.flank_size + len(allele1) : self.flank_size * 2 + mismatch_length
            ]

            allele1_seq = left_flank + allele1 + allele1_right_flank
            allele2_seq = left_flank + allele2 + allele2_right_flank

        assert len(allele1_seq) == self.flank_size * 2
        assert len(allele2_seq) == self.flank_size * 2

        allele1_coords, allele2_coords = [chrom, -1, -1], [chrom, -1, -1]
        # we don't support coords for indel variants
        if self.return_coords and (len(allele1) == len(allele2)):
            allele1_coords = [chrom, pos - self.flank_size, pos + self.flank_size]
            allele2_coords = allele1_coords
            # validate
            assert allele1_coords[2] - allele1_coords[1] == len(allele1_seq)
            assert allele2_coords[2] - allele2_coords[1] == len(allele2_seq)
            seq = str(self.genome[chrom][allele1_coords[1] : allele1_coords[2]])
            # seq could equal allele1_seq or allele2_seq depending on which is the reference allele
            # or it could equal neither, if the variants stem from a study where neither allele is a reference allele
            # in the latter case, emit a Warning, which could be suppressed if desired
            if (seq != allele1_seq) and (seq != allele2_seq):
                msg = "Neither allele seems to be a reference allele.\n" \
                    f"chrom: {chrom}, pos: {pos}, allele1: {allele1}, allele2: {allele2}"
                warnings.warn(msg)

        return allele1_seq, allele2_seq, allele1_coords, allele2_coords

    def __getitem__(self, idx):
        cur_entries = self.variants_table.iloc[
            idx
            * self.batch_size : min([self.num_variants, (idx + 1) * self.batch_size])
        ]
        rsids = cur_entries["rsid"].tolist()

        if self.shuf:
            allele1_seqs, allele2_seqs, allele1_coords, allele2_coords = zip(
                *[
                    self.__get_allele_seq__(v, w, x, y, z)
                    for v, w, x, y, z in zip(
                        cur_entries.chr,
                        cur_entries.pos,
                        cur_entries.allele1,
                        cur_entries.allele2,
                        cur_entries.random_seed,
                    )
                ]
            )
        else:
            allele1_seqs, allele2_seqs, allele1_coords, allele2_coords = zip(
                *[
                    self.__get_allele_seq__(w, x, y, z)
                    for w, x, y, z in zip(
                        cur_entries.chr,
                        cur_entries.pos,
                        cur_entries.allele1,
                        cur_entries.allele2,
                    )
                ]
            )

        allele1_seqs_ret = (
            list(allele1_seqs)
            if self.debug_mode
            else one_hot.dna_to_one_hot(list(allele1_seqs))
        )
        allele2_seqs_ret = (
            list(allele2_seqs)
            if self.debug_mode
            else one_hot.dna_to_one_hot(list(allele2_seqs))
        )

        if not self.return_coords:
            return (
                rsids,
                allele1_seqs_ret,
                allele2_seqs_ret,
            )
        return (
            rsids,
            allele1_seqs_ret,
            allele2_seqs_ret,
            list(allele1_coords),
            list(allele2_coords),
        )

    def __len__(self):
        return math.ceil(self.num_variants / self.batch_size)
