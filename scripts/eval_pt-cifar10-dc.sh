#! /bin/bash
CUDA_VISIBLE_DEVICES=0 python3 eval_pt.py \
--dataset CIFAR10 \
--model dc32 \
--z_dim 128 \
--begin 138000 \
--end 266000 \
--step 2000 \
--model_dir checkpoints/cifar-BCGD/BCGD-dc320.010_ \
--logdir eval_results/cifar10-dc32BCGD/ \
--eval_fid
#--eval_is
