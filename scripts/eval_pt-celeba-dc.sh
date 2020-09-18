#! /bin/bash
CUDA_VISIBLE_DEVICES=2 python3 eval_pt.py \
--dataset CelebA \
--model DCGAN \
--z_dim 100 \
--dim 3 \
--begin 26000 \
--end 36500 \
--step 500 \
--model_dir checkpoints/celeba-SCG/SCG-DCGAN0.000_ \
--logdir eval_results/celeba-SCG/ \
--eval_fid
