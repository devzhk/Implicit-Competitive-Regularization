#! /bin/bash
CUDA_VISIBLE_DEVICES=0,1 python3 train_stylegan_cgd.py \
--batch 32 \
--size 128 \
--iter 100000 \
--path /mnt/md1/visiondatasets/datas/ffhq/ffhq128.mdb \
--wandb \
--optimizer ACGD \
--lr_d 1e-3 \
--lr_g 2e-3 \
--gpu_num 2 \
--tol 1e-4 \
--ckpt checkpoints/stylegan-acgd/002000.pt
