#!/bin/bash
#SBATCH --account=bebv-delta-gpu
#SBATCH --job-name=sbatch_build_pytorch
#SBATCH --output=sbatch_build_pytorch_%j.out
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=0
#SBATCH --time=08:00:00

set -x

cd $SLURM_SUBMIT_DIR
apptainer build --force pytorch.sif pytorch.def
