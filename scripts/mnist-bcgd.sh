#! /bin/bash
CUDA_VISIBLE_DEVICES=1 python3 train_simul.py \
--gpu_num 1 \
--epoch_num 100 \
--model mnist \
--z_dim 96 \
--optimizer BCGD \
--momentum 0.7 \
--tol 1e-8 \
--atol 1e-12 \
--loss_type WGAN \
--show_iter 500 \
--logdir mnist-BCGD7 \
--dataset MNIST \
--datapath /mnt/md1/visiondatasets/datas/mnist \
--lr_d 1e-1 \
--lr_g 1e-1 \
--batchsize 128
#--startn 46500 \
#--checkpoint checkpoints/mnist-BCGD/BCGD-mnist0.010_46500.pth