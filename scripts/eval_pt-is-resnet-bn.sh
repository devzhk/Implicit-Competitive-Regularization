#! /bin/bash
CUDA_VISIBLE_DEVICES=2 python3.6 eval_pt.py \
--model ResGAN \
--z_dim 128 \
--begin 640000 \
--end 1310000 \
--step 5000 \
--model_dir /home/shehuajun/lsy/hongkai/rebuttal/CIFAR10-0.00010/wgan-0.00010_ \
--logdir /home/shehuajun/lsy/hongkai/eval_results/cifar-adam/ \
--eval_is