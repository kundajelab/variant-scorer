# Av's code with a bit of reformatting
# Adapted from Zahoor's mtbatchgen

from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import load_model
import tensorflow as tf
import scipy.stats
from scipy.spatial.distance import jensenshannon
import pandas as pd
import os
import argparse
import numpy as np
import h5py
import math
from tqdm import tqdm
import sys
sys.path.append('..')
from generators.variant_generator import SNPGenerator
from generators.peak_generator import PeakGenerator
from utils import argmanager, losses
import shap
from deeplift.dinuc_shuffle import dinuc_shuffle
tf.compat.v1.disable_v2_behavior()


def combine_mult_and_diffref(mult, orig_inp, bg_data):
    to_return = []
    
    for l in [0]:
        projected_hypothetical_contribs = \
            np.zeros_like(bg_data[l]).astype("float")
        assert len(orig_inp[l].shape)==2
        
        # At each position in the input sequence, we iterate over the
        # one-hot encoding possibilities (eg: for genomic sequence, 
        # this is ACGT i.e. 1000, 0100, 0010 and 0001) and compute the
        # hypothetical difference-from-reference in each case. We then 
        # multiply the hypothetical differences-from-reference with 
        # the multipliers to get the hypothetical contributions. For 
        # each of the one-hot encoding possibilities, the hypothetical
        # contributions are then summed across the ACGT axis to 
        # estimate the total hypothetical contribution of each 
        # position. This per-position hypothetical contribution is then
        # assigned ("projected") onto whichever base was present in the
        # hypothetical sequence. The reason this is a fast estimate of
        # what the importance scores *would* look like if different 
        # bases were present in the underlying sequence is that the
        # multipliers are computed once using the original sequence, 
        # and are not computed again for each hypothetical sequence.
        for i in range(orig_inp[l].shape[-1]):
            hypothetical_input = np.zeros_like(orig_inp[l]).astype("float")
            hypothetical_input[:, i] = 1.0
            hypothetical_difference_from_reference = \
                (hypothetical_input[None, :, :] - bg_data[l])
            hypothetical_contribs = hypothetical_difference_from_reference * \
                                    mult[l]
            projected_hypothetical_contribs[:, :, i] = \
                np.sum(hypothetical_contribs, axis=-1) 
            
        to_return.append(np.mean(projected_hypothetical_contribs,axis=0))

    if len(orig_inp)>1:
        to_return.append(np.zeros_like(orig_inp[1]))
    
    return to_return


def shuffle_several_times(s):
    numshuffles=20
    if len(s)==2:
        return [np.array([dinuc_shuffle(s[0]) for i in range(numshuffles)]),
                np.array([s[1] for i in range(numshuffles)])]
    else:
        return [np.array([dinuc_shuffle(s[0]) for i in range(numshuffles)])]


def get_weightedsum_meannormed_logits(model):
    # See Google slide deck for explanations
    # We meannorm as per section titled 
    # "Adjustments for Softmax Layers" in the DeepLIFT paper
    meannormed_logits = (model.outputs[0] - \
                         tf.reduce_mean(model.outputs[0], axis=1)[:, None])

    # 'stop_gradient' will prevent importance from being propagated
    # through this operation; we do this because we just want to treat
    # the post-softmax probabilities as 'weights' on the different 
    # logits, without having the network explain how the probabilities
    # themselves were derived. Could be worth contrasting explanations
    # derived with and without stop_gradient enabled...
    stopgrad_meannormed_logits = tf.stop_gradient(meannormed_logits)
    softmax_out = tf.nn.softmax(stopgrad_meannormed_logits, axis=1)
    
    # Weight the logits according to the softmax probabilities, take
    # the sum for each example. This mirrors what was done for the
    # bpnet paper.
    weightedsum_meannormed_logits = tf.reduce_sum(softmax_out * \
                                                  meannormed_logits,
                                                  axis=1)
    
    return weightedsum_meannormed_logits


def fetch_shap(model, variants_table, input_len, genome_fasta, batch_size, debug_mode=False, lite=False, bias=None, shuf=False):
    rsids = []
    allele1_counts_shap = []
    allele2_counts_shap = []

    # variant sequence generator
    var_gen = SNPGenerator(variants_table=variants_table,
                           input_len=input_len,
                           genome_fasta=genome_fasta,
                           batch_size=batch_size,
                           debug_mode=debug_mode,
                           shuf=shuf)

    for i in tqdm(range(len(var_gen))):

        batch_rsids, allele1_seqs, allele2_seqs = var_gen[i]

        if lite:
            counts_model_input = [model.input[0], model.input[2]]
            allele1_input = [allele1_seqs, np.zeros((allele1_seqs.shape[0], 1))]
            allele2_input = [allele2_seqs, np.zeros((allele2_seqs.shape[0], 1))]

            profile_model_counts_explainer = shap.explainers.deep.TFDeepExplainer(
                (counts_model_input, tf.reduce_sum(model.outputs[1], axis=-1)),
                shuffle_several_times,
                combine_mult_and_diffref=combine_mult_and_diffref)

            allele1_counts_shap_batch = profile_model_counts_explainer.shap_values(
                allele1_input, progress_message=10)
            allele2_counts_shap_batch = profile_model_counts_explainer.shap_values(
                allele2_input, progress_message=10)

            allele1_counts_shap_batch = allele1_counts_shap_batch[0] * allele1_input[0]
            allele2_counts_shap_batch = allele2_counts_shap_batch[0] * allele2_input[0]

        else:
            counts_model_input = model.input
            allele1_input = allele1_seqs
            allele2_input = allele2_seqs

            profile_model_counts_explainer = shap.explainers.deep.TFDeepExplainer(
                (counts_model_input, tf.reduce_sum(model.outputs[1], axis=-1)),
                shuffle_several_times,
                combine_mult_and_diffref=combine_mult_and_diffref)

            allele1_counts_shap_batch = profile_model_counts_explainer.shap_values(
                allele1_input, progress_message=10)
            allele2_counts_shap_batch = profile_model_counts_explainer.shap_values(
                allele2_input, progress_message=10)

            allele1_counts_shap_batch = allele1_counts_shap_batch * allele1_input
            allele2_counts_shap_batch = allele2_counts_shap_batch * allele2_input

        allele1_counts_shap.extend(allele1_counts_shap_batch)
        allele2_counts_shap.extend(allele2_counts_shap_batch)

        rsids.extend(batch_rsids)

    return np.array(rsids), np.array(allele1_counts_shap), np.array(allele2_counts_shap)

