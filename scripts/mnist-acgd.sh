#! /bin/bash
CUDA_VISIBLE_DEVICES=1 python3 train_simul.py \
--gpu_num 1 \
--epoch_num 100 \
--model mnist \
--z_dim 96 \
--optimizer ACGD \
--loss_type WGAN \
--show_iter 500 \
--logdir mnist-ACGD \
--dataset MNIST \
--datapath mnist \
--lr_d 1e-4 \
--lr_g 1e-4 \
--batchsize 128