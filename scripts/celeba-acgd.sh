#! /bin/bash
CUDA_VISIBLE_DEVICES=1 python3 train_simul.py \
--gpu_num 1 \
--epoch_num 15 \
--model DCGAN \
--z_dim 100 \
--optimizer ACGD \
--momentum 0.0 \
--loss_type WGAN \
--show_iter 500 \
--logdir celeba-ACGD \
--dataset CelebA \
--datapath /mnt/md1/visiondatasets/datas/celeba \
--lr_d 2e-4 \
--lr_g 2e-4 \
--batchsize 128 \
--tol 1e-8 \
--atol 1e-12