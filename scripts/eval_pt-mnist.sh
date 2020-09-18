#! /bin/bash
CUDA_VISIBLE_DEVICES=1 python3 eval_pt.py \
--dataset MNIST \
--model mnist \
--z_dim 96 \
--begin 38000 \
--end 46500 \
--step 500 \
--model_dir checkpoints/mnist-BCGD1/BCGD-mnist0.010_ \
--logdir eval_results/mnist-BCGD1/ \
--eval_fid