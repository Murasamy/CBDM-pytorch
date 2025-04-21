CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --train  \
        --flagfile ./config/cifar100.txt --parallel \
        --logdir ./logs/cifar100lt_ddpm --total_steps 300001 \
        --edm2_truncate True --edm2_truncate_portion 0.7 \
        --conditional \
        --data_type cifar100lt --imb_factor 0.01 --img_size 32 \
        --batch_size 64 --save_step 100000 --sample_step 50000 \
        --cfg