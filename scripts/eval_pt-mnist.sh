#! /bin/bash
CUDA_VISIBLE_DEVICES=3 python3 eval_pt.py \
--dataset MNIST \
--model mnist \
--z_dim 96 \
--begin 0 \
--end 46500 \
--step 500 \
--model_dir checkpoints/mnist-BCGD3/BCGD-mnist0.010_ \
--logdir eval_results/mnist-BCGD3/ \
--eval_fid