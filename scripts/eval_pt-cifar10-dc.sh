#! /bin/bash
CUDA_VISIBLE_DEVICES=3 python3 eval_pt.py \
--dataset CIFAR10 \
--model dc \
--z_dim 128 \
--begin 305000 \
--end 405000 \
--step 5000 \
--model_dir checkpoints/cifar10-SCG/SCG-dc0.000_ \
--logdir eval_results/cifar10-SCG/ \
--eval_is --eval_fid
