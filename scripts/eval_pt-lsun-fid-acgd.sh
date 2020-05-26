#! /bin/bash
CUDA_VISIBLE_DEVICES=1 python3 eval_pt.py \
--dataset lsun-bedroom \
--model DCGAN \
--z_dim 100 \
--begin 229000 \
--end 236000 \
--step 1000 \
--model_dir checkpoints/ACGD/ACGD-DCGAN0.010_ \
--logdir eval_results/lsunbedroom-ACGD/ \
--eval_fid