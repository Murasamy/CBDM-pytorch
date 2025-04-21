conda activate opence-v1.6.1;

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --train  \
        --flagfile ./config/cifar100.txt --parallel \
        --logdir ./logs/cifar100lt_seperate_umsampler \
        --seperate_upsampler True \
        --seperate_unconditional_step 200 \
        --seperate_upsampler_step 200 \
        --conditional \
        --data_type cifar100lt --imb_factor 0.01 --img_size 32 \
        --batch_size 64 --save_step 100000 --sample_step 50000 \
        --cfg