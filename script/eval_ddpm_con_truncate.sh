python main.py \
    --flagfile ./logs/cifar100lt_ddpm/flagfile.txt \
    --logdir ./logs/cifar100lt_ddpm_truncate \
    --fid_cache ./stats/cifar100.train.npz \
    --edm2_truncate True --0.7 \
    --ckpt_step 200000 \
    --num_images 50000 --batch_size 64 \
    --notrain \
    --eval \
    --sample_method cfg  --omega 0.8
