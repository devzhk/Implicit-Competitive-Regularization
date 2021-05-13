#! /bin/bash
CUDA_VISIBLE_DEVICES=0 python3.6 calculate_stats.py \
--dataset FFHQ \
--data_path /mnt/md1/visiondatasets/datas/ffhq/ffhq128.mdb \
--image_size 128