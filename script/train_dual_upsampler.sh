python main.py --train  \
        --flagfile ./config/cifar100.txt --parallel \
        --logdir ./logs/cifar100lt_dual_sampler --total_steps 300001 \
        --conditional \
        --data_type cifar100lt --imb_factor 0.01 --img_size 32 \
        --batch_size 64 --save_step 100000 --sample_step 50000 \
        --cfg \
        --shared_portion 0.1