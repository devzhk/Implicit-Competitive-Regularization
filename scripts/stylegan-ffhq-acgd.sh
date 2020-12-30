#! /bin/bash
CUDA_VISIBLE_DEVICES=0,2 python3 train_stylegan_cgd.py \
--batch 8 \
--size 128 \
--path /mnt/md1/visiondatasets/datas/ffhq/ffhq128.mdb \
--wandb \
--optimizer ACGD \
--lr_d 1e-4 \
--lr_g 1e-4 \
--gpu_num 2 \
--tol 1e-4