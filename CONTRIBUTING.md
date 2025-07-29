
## Test data

Test data is derived from the Africa caQTLs associated with the ChromBPNet preprint
(Pampari et al, biorxiv 2024). The variants are derived from the dataset
on Synapse at https://www.synapse.org/Synapse:syn64126781.

Download and processing of these variants to prepare the test data is documented
at `scripts/get_caqtl_data.sh`.


## Unit testing

Unit testing is set up with `pytest`.

Some of the tests depend on ChromBPNet models or genome references stored on Oak at
`${OAK}/projects/variant-scorer-test`.

For example, to run the tests on Sherlock, request an interactive node with a GPU:

```bash
sh_dev -g 1 -t 120
```

Activate your associated conda environment for the `variant-scorer` repo. (Install `pytest`
there if needed.)

```bash
conda activate variant-scorer
pip install pytest
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