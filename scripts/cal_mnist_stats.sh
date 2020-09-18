#! /bin/bash
CUDA_VISIBLE_DEVICES=1 python3.6 calculate_stats.py \
--dataset MNIST \
--data_path ../datas/mnist