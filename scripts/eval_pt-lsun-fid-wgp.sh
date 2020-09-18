#! /bin/bash
CUDA_VISIBLE_DEVICES=1 python3.6 eval_pt.py \
--dataset LSUN-bedroom \
--model DCGAN \
--z_dim 100 \
--begin 0 \
--end 1010000 \
--step 10000 \
--model_dir rebuttal/LSUN-bedroom-0.00010/DCGAN-GP-n-0.00010_ \
--logdir eval_results/lsunbedroom-WGP/ \
--eval_fid