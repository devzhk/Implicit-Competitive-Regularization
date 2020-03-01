#! /bin/bash
CUDA_VISIBLE_DEVICES=0,1 python3 VisionData.py \
--gpu_num 2 \
--optimizer ACGD \
--batchsize 64 \
--logdir ACGD \
--dataset CIFAR \
--z_dim 128 \
--loss_type WGAN \
--lr_d 1e-4 \
--eval_is \
--show_iter 1000 \
--epoch_num 600
