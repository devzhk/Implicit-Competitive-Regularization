#! /bin/bash
CUDA_VISIBLE_DEVICES=2 python3.6 wgan_gp.py \
--gpu_num 1 \
--epoch_num 3000 \
--model dc \
--loss_type WGAN \
--z_dim 128 \
--show_iter 5000 \
--logdir dcGAN-WGP \
--dataset CIFAR10 \
--datapath cifar10 \
--lr_d 1e-4 \
--lr_g 1e-4 \
--batchsize 64 \
--gp_weight 10 \
--d_iter 5