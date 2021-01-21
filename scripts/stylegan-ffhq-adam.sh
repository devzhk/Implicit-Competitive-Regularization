#! /bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch \
--nproc_per_node=4 --master_port=7600 \
train_stylegan.py \
--iter 200000 \
--size 128 \
--batch 8 \
--path /mnt/md1/visiondatasets/datas/ffhq/ffhq128.mdb \
--wandb