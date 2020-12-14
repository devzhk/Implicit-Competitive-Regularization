#! /bin/bash
CUDA_VISIBLE_DEVICES=2 python3 eval_pt.py \
--dataset CIFAR10 \
--model dc \
--z_dim 128 \
--begin 0 \
--end 465000 \
--step 5000 \
--model_dir checkpoints/cifar-ACGD/ACGD-dc0.010_ \
--logdir eval_results/cifar10-dcACGD-old/ \
--eval_fid
#--eval_is
