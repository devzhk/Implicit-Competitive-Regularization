#! /bin/bash
CUDA_VISIBLE_DEVICES=3 python3.6 wgan_gp.py \
--gpu_num 1 \
--epoch_num 1000 \
--model ResGAN \
--weight_path rebuttal/CIFAR10-0.00010/wgan-0.00010_630000.pth \
--startPoint 630000 \
--loss_type WGAN \
--z_dim 128 \
--show_iter 1000 \
--logdir ResGAN-GP \
--dataset CIFAR10 \
--lr_d 1e-4 \
--lr_g 1e-4 \
--batchsize 64 \
--gp_weight 10 \
--d_iter 5