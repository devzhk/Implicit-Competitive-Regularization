#! /bin/bash
CUDA_VISIBLE_DEVICES=0,1 python3 train_stylegan_cgd.py \
--batch 16 \
--size 128 \
--iter 100000 \
--path /mnt/md1/visiondatasets/datas/ffhq/ffhq128.mdb \
--wandb \
--optimizer ACGD \
--lr_d 1e-5 \
--lr_g 1e-4 \
--gpu_num 2 \
--tol 1e-4