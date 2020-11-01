#! /bin/bash
CUDA_VISIBLE_DEVICES=1,2 python3 train_simul.py \
--gpu_num 2 \
--epoch_num 15 \
--model DCGANs \
--z_dim 100 \
--optimizer ACGD \
--momentum 0.0 \
--loss_type WGAN \
--show_iter 2500 \
--logdir lsun-bedroom-DC256ACGD \
--dataset LSUN-bedroom \
--datapath /mnt/md1/visiondatasets/datas/lsun \
--lr_d 1e-4 \
--lr_g 1e-4 \
--batchsize 128 \
--tol 1e-8 \
--atol 1e-12 \
--model_config scripts/configs/DCGAN.yaml
