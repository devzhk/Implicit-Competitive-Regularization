#! /bin/bash
CUDA_VISIBLE_DEVICES=2,0 python3 cgd_stochastic.py \
--gpu_num 2 \
--epoch_num 600 \
--model dc \
--z_dim 128 \
--optimizer SCG \
--loss_type WGAN \
--show_iter 5000 \
--logdir cifar10-SCG \
--dataset CIFAR10 \
--datapath cifar10 \
--lr_d 1e-4 \
--lr_g 1e-4 \
--batchsize 64 \
--strategy scg \
--tol 1e-10 \
--atol 1e-14