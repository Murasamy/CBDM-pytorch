# only use cuda1
# CUDA_VISIBLE_DEVICES=1 \
python main_eval.py \
    --flagfile ./logs/cifar100lt_ddpm_cond/flagfile.txt \
    --logdir ./logs/cifar100lt_ddpm_cond \
    --fid_cache ./stats/cifar100.train.npz \
    --ckpt_step 200000 \
    --num_images 50000 --batch_size 512 \
    --notrain \
    --eval \
    --same_label True \
    --sample_method cfg  --omega 0 \
    --data_type cifar100lt

