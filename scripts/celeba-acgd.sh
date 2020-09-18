#! /bin/bash
CUDA_VISIBLE_DEVICES=2 python3 train_simul.py \
--gpu_num 1 \
--epoch_num 15 \
--model DCGAN \
--z_dim 100 \
--optimizer ACGD \
--loss_type WGAN \
--show_iter 500 \
--logdir celeba-acgd \
--dataset CelebA \
--datapath celeba \
--lr_d 2e-4 \
--lr_g 2e-4 \
--batchsize 128 \
--tol 1e-10 \
--atol 1e-14