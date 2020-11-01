#! /bin/bash
CUDA_VISIBLE_DEVICES=2 python3 eval_pt.py \
--dataset CelebA \
--model DCGAN \
--z_dim 100 \
--dim 3 \
--begin 0 \
--end 23500 \
--step 500 \
--model_dir checkpoints/celeba-Adam/Adam-DCGAN_ \
--logdir eval_results/celeba-Adam/ \
--eval_fid
