#! /bin/bash
CUDA_VISIBLE_DEVICES=1 python3.6 eval_pt.py \
--dataset cifar10 \
--model ResGAN \
--z_dim 128 \
--begin 96000 \
--end 165000 \
--step 1000 \
--model_dir /home/shehuajun/lsy/hongkai/checkpoints/ACGD/ACGD-Resnet0.010_ \
--logdir /home/shehuajun/lsy/hongkai/eval_results/cifar-ACGD/ \
--eval_fid