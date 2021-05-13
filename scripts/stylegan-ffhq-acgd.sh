#! /bin/bash
CUDA_VISIBLE_DEVICES=0,2 python3 train_stylegan_cgd.py \
--batch 32 \
--size 128 \
--iter 200000 \
--path /mnt/md1/visiondatasets/datas/ffhq/ffhq128.mdb \
--wandb \
--optimizer ACGD \
--lr_d 2e-3 \
--lr_g 2e-3 \
--gpu_num 2 \
--tol 1e-4 \
--ckpt checkpoints/stylegan/190000.pt
