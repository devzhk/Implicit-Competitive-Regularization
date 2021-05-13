#! /bin/bash
CUDA_VISIBLE_DEVICES=2 python3 eval_pt.py \
--dataset CelebA \
--model DCGAN \
--z_dim 100 \
--dim 3 \
--begin 20000 \
--end 22000 \
--step 100 \
--model_dir checkpoints/celeba-adam-restart/Adam-DCGAN_ \
--logdir eval_results/celeba-Adam-restart/ \
--eval_fid
