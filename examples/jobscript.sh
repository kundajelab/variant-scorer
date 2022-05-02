#!/bin/bash

module load cudnn

# https://stackoverflow.com/questions/34534513/calling-conda-source-activate-from-bash-script
eval "$(conda shell.bash hook)"
conda activate chrombpnet

echo "Live"
python "$@"

