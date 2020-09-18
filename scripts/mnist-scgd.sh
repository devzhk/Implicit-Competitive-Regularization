#! /bin/bash
CUDA_VISIBLE_DEVICES=1 python3 cgd_stochastic.py \
--gpu_num 1 \
--epoch_num 50 \
--model mnist \
--z_dim 96 \
--optimizer SCG \
--loss_type WGAN \
--show_iter 10 \
--logdir mnist-SCG \
--dataset MNIST \
--datapath mnist \
--lr_d 1e-4 \
--lr_g 1e-4 \
--batchsize 128 \
--strategy scg