#! /bin/bash
CUDA_VISIBLE_DEVICES=3 python3 eval_pt.py \
--dataset cifar10 \
--model dc \
--z_dim 128 \
--begin 350000 \
--end 465000 \
--step 5000 \
--model_dir checkpoints/ACGD64/BCGD-dc0.010_ \
--logdir eval_results/cifar-bcgd64/ \
--eval_is --eval_fid
