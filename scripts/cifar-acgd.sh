#! /bin/bash
CUDA_VISIBLE_DEVICES=2,1 python3 train_simul.py \
--gpu_num 2 \
--epoch_num 6000 \
--model dc \
--z_dim 128 \
--optimizer ACGD \
--loss_type WGAN \
--show_iter 25000 \
--logdir ACGD1024 \
--dataset CIFAR10 \
--datapath cifar10 \
--lr_d 1e-4 \
--lr_g 1e-4 \
--batchsize 1024 \
--tol 1e-10 \
--atol 1e-14