#! /bin/bash
CUDA_VISIBLE_DEVICES=1 python3.6 calculate_stats.py \
--dataset CIFAR10 \
--data_path ../datas/cifar10