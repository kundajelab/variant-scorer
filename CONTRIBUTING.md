
## Test data

Test data is derived from the Africa caQTLs associated with the ChromBPNet preprint
(Pampari et al, biorxiv 2024). The variants are derived from the dataset
on Synapse at https://www.synapse.org/Synapse:syn64126781.

Download and processing of these variants to prepare the test data is documented
at `scripts/get_caqtl_data.sh`.


## Environment setup

Create the conda environment from the provided spec:

```bash
conda env create -f environment.yml
conda activate variant-scorer
```

For GPU support (required for running models), replace the default PyTorch install with
the CUDA-enabled build matching your driver. For example, for CUDA 12.1:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

See https://pytorch.org/get-started/locally/ for the right command for your system.


## Unit testing

Unit testing is set up with `pytest`.

Some of the tests depend on ChromBPNet models or genome references stored on Oak at
`${OAK}/projects/variant-scorer-test`.

For example, to run the tests on Sherlock, request an interactive node with a GPU:

```bash
sh_dev -g 1 -t 120
```

Activate the conda environment:

```bash
conda activate variant-scorer
```

Check the output of your `OAK` variable:

```bash
echo $OAK
```

Run the tests:

```bash
pytest -rs -s
```

Optionally, to run without a GPU, use:

```bash
pytest -rs -s -m "not gpu"
```

Or, to skip all the tests that require Oak data, use:

```bash
pytest -rs -s -m "not oak"
```

Or, run a specific test:

```bash
pytest -rs -s tests/test_variant_scoring.py::TestVariantScoringCLI::test_variant_scoring_per_chrom
```