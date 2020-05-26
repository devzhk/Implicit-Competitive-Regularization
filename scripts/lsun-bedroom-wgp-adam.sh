#! /bin/bash
CUDA_VISIBLE_DEVICES=1,3 python3 wgan_gp.py \
--gpu_num 2 \
--dataset LSUN-bedroom \
--datapath lsun \
--epoch_num 1000 \
--model DCGAN \
--startPoint 0 \
--loss_type WGAN \
--z_dim 100 \
--show_iter 1000 \
--logdir LSUN_DCGAN-GP \
--lr_d 1e-4 \
--lr_g 1e-4 \
--batchsize 64 \
--gp_weight 10 \
--d_iter 5