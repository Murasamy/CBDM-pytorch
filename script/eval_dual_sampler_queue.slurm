#!/bin/bash
#SBATCH --job-name="cbdm_queue"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4 
#SBATCH --output="/home/pwu3/CBDM-pytorch/slurmOut/a.out.%j.%N.out"
#SBATCH --error="/home/pwu3/CBDM-pytorch/slurmOut/%j_%x.err"
#SBATCH --partition=gpu
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:v100:4
#SBATCH --mail-user=wpy3167363942@gmail.com
#SBATCH --mail-type=ALL
module purge;
conda activate opence-v1.6.1;
python main_dual_upsampler.py \
    --flagfile ./logs/cifar100lt_dual_sampler03/flagfile.txt \
    --logdir ./logs/cifar100lt_dual_sampler03 \
    --fid_cache ./stats/cifar100.train.npz \
    --ckpt_step 100000 \
    --num_images 50000 --batch_size 1024 \
    --notrain \
    --eval \
    --sample_method cfg  --omega 0.65

