#! /bin/bash
CUDA_VISIBLE_DEVICES=1 python3 eval_pt.py \
--dataset LSUN-bedroom \
--model DCGAN \
--z_dim 100 \
--dim 3 \
--begin 0 \
--end 355000 \
--step 2500 \
--model_dir checkpoints/lsun-bedroom-ACGD3/ACGD-DCGAN0.010_ \
--logdir eval_results/lsunbedroom-ACGD3/ \
--eval_fid