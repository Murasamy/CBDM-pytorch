#!/bin/bash

#SBATCH --job-name="cbdm_queue"
#SBATCH --output="/home/pwu3/CBDM-pytorch/slurmOut/a.out.%j.%N.out"
#SBATCH --error="/home/pwu3/CBDM-pytorch/slurmOut%j_%x.err"
#SBATCH --partition=gpux4
#SBATCH --cpu_per_gpu=16
#SBATCH --time=24
#SBATCH --mail-user=
#SBATCH --mail-type=ALL

module purge;
conda activate opence-v1.6.1;

