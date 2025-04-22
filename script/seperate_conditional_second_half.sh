# unset  LD_LIBRARY_PATH;

CUDA_VISIBLE_DEVICES=0,1,2,3 python main_second_half.py --train  \
        --flagfile ./config/cifar100.txt --parallel \
        --logdir ./logs/cifar100lt_seperate_umsampler \
        --seperate_upsampler True \
        --freeze_down_latent_label False \
        --ckpt_step 200004 \
        --seperate_unconditional_step 200003 \
        --seperate_upsampler_step 100003 \
        --conditional \
        --data_type cifar100lt --imb_factor 0.01 --img_size 32 \
        --batch_size 64 --save_step 100000 --sample_step 50000 \
        --cfg