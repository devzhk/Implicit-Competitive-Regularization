#! /bin/bash
CUDA_VISIBLE_DEVICES=3,0 python3 cgd_stochastic.py \
--gpu_num 2 \
--epoch_num 15 \
--model DCGAN \
--checkpoint checkpoints/celeba-SCG/SCG-DCGAN0.000_15000.pth \
--startn 15000 \
--z_dim 100 \
--optimizer SCG \
--loss_type WGAN \
--show_iter 500 \
--logdir celeba-SCG \
--dataset CelebA \
--datapath celeba \
--lr_d 1e-4 \
--lr_g 1e-4 \
--batchsize 128 \
--strategy scg \
--tol 1e-10 \
--atol 1e-14