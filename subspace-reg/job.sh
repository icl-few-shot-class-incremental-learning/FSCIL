#!/bin/bash

echo "### START DATE=$(date)"
echo "### HOSTNAME=$(hostname)"
echo "### CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# conda 환경 활성화.
source  ~/.bashrc
conda activate subreg

# cuda 11.0 환경 구성.
ml purge
ml load cuda/11.0

# 활성화된 환경에서 코드 실행.
bash scripts/continual/slurm_semantic_subspace_reg.sh

echo "###"
echo "### END DATE=$(date)"
