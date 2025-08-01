import pytest
import os


@pytest.fixture(scope="session")
def test_data_dir():
    return os.path.join(os.path.dirname(__file__), 'data')


@pytest.fixture(scope="session")
def out_dir():
	"""Fixture to provide output directory for tests"""
	return os.path.join(os.path.dirname(__file__), 'outputs')


@pytest.fixture(scope="session")
def src():
    return os.path.join(os.path.dirname(__file__), '..', 'src')


@pytest.fixture(scope="session")
def oak_path():
	"""Get OAK path from environment variable or skip test"""
	oak = os.path.join(os.environ.get('OAK'), "projects/variant-scorer-test")
	if oak is None:
		pytest.skip("OAK environment variable not set")
	if not os.path.exists(oak):
		pytest.skip(f"OAK file not found: {oak}")
	return oak


@pytest.fixture(scope="session")
def genome_path(oak_path):
    """Get genome path from environment variable or skip test"""
    genome = os.path.join(oak_path, "GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta")
    if genome is None:
        pytest.skip("Genome path not set")
    if not os.path.exists(genome):
        pytest.skip(f"Genome file not found: {genome}")
    return genome


@pytest.fixture(scope="session")
def model_paths(oak_path):
    """Get model path from environment variable or skip test"""
    model = [os.path.join(oak_path, f"GM12878_DNase_models/fold_{i}/model.chrombpnet_nobias.fold_{i}.ENCSR000EMT.h5") for i in range(5)]
    for m in model:
        if not os.path.exists(m):
            pytest.skip(f"Model file not found: {m}")
    return model


@pytest.fixture(scope="session")
def chrom_sizes_path(oak_path):
    """Get chrom sizes path from environment variable or skip test"""
    chrom_sizes = os.path.join(oak_path, "GRCh38_EBV_sorted_standard.chrom.sizes.tsv")
    if chrom_sizes is None:
        pytest.skip("Chrom sizes path not set")
    if not os.path.exists(chrom_sizes):
        pytest.skip(f"Chrom sizes file not found: {chrom_sizes}")
    return chrom_sizes
