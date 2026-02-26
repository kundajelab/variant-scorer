import numpy as np
import torch
from tqdm import tqdm
import sys
sys.path.append('..')
from generators.variant_generator import VariantGenerator
from bpnetlite.bpnet import CountWrapper, ProfileWrapper
from bpnetlite.attribute import deep_lift_shap


def fetch_shap(model, variants_table, input_len, genome_fasta, batch_size,
               debug_mode=False, lite=False, bias=None, shuf=False,
               shap_type="counts"):
    """Compute DeepLIFT-SHAP contribution scores for variant sequences.

    Parameters
    ----------
    model: torch.nn.Module
        A BPNet model loaded via BPNet.from_chrombpnet(). Will be wrapped
        with CountWrapper (counts SHAP) or ProfileWrapper (profile SHAP).
    variants_table: pd.DataFrame
    input_len: int
    genome_fasta: str
    batch_size: int
    shap_type: str
        'counts' or 'profile'

    Returns
    -------
    variant_ids: np.ndarray, shape (N,)
    allele1_inputs: np.ndarray, shape (N, 4, L)
    allele2_inputs: np.ndarray, shape (N, 4, L)
    allele1_shap: np.ndarray, shape (N, 4, L)
    allele2_shap: np.ndarray, shape (N, 4, L)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if shap_type == "counts":
        wrapped = CountWrapper(model).eval().to(device)
    else:
        assert shap_type == "profile"
        wrapped = ProfileWrapper(model).eval().to(device)

    var_gen = VariantGenerator(variants_table=variants_table,
                               input_len=input_len,
                               genome_fasta=genome_fasta,
                               batch_size=batch_size,
                               debug_mode=False,
                               shuf=shuf)

    variant_ids = []
    allele1_inputs = []
    allele2_inputs = []
    allele1_shap_list = []
    allele2_shap_list = []

    for i in tqdm(range(len(var_gen))):
        batch_variant_ids, allele1_seqs, allele2_seqs = var_gen[i]
        # seqs shape: (N, 4, L)

        allele1_tensor = torch.tensor(allele1_seqs, dtype=torch.float32)
        allele2_tensor = torch.tensor(allele2_seqs, dtype=torch.float32)

        allele1_attr = deep_lift_shap(wrapped, allele1_tensor, random_state=0,
                                      device=device)
        allele2_attr = deep_lift_shap(wrapped, allele2_tensor, random_state=0,
                                      device=device)

        # projected SHAP: hypothetical scores multiplied by the one-hot input
        allele1_projected = (allele1_attr * allele1_tensor).cpu().numpy()
        allele2_projected = (allele2_attr * allele2_tensor).cpu().numpy()

        allele1_inputs.extend(allele1_seqs)
        allele2_inputs.extend(allele2_seqs)
        allele1_shap_list.extend(allele1_projected)
        allele2_shap_list.extend(allele2_projected)
        variant_ids.extend(batch_variant_ids)

    return (np.array(variant_ids),
            np.array(allele1_inputs),
            np.array(allele2_inputs),
            np.array(allele1_shap_list),
            np.array(allele2_shap_list))
