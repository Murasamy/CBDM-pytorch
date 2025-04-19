export PATH="/u/pwu3/miniconda3/envs/python6/bin:/u/pwu3/miniconda3/bin:/u/pwu3/miniconda3/condabin:$PATH"
source ~/miniconda3/etc/profile.d/conda.sh;
conda activate python6;
wandb login 521eacb01d9cd4e083cf5a065774992dd857a527;

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --train  \
        --flagfile ./config/cifar100.txt --parallel \
        --logdir ./logs/cifar100lt_ddpm --total_steps 300001 \
        --conditional \
        --data_type cifar100lt --imb_factor 0.01 --img_size 32 \
        --batch_size 64 --save_step 100000 --sample_step 50000 \
        --cfg