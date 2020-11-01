#! /bin/bash
CUDA_VISIBLE_DEVICES=1 python3 train_seq.py \
--gpu_num 1 \
--epoch_num 15 \
--model DCGAN \
--z_dim 100 \
--optimizer Adam \
--loss_type JSD \
--show_iter 500 \
--logdir celeba-adam \
--dataset CelebA \
--datapath /mnt/md1/visiondatasets/datas/celeba \
--lr_d 2e-4 \
--lr_g 2e-4 \
--batchsize 128