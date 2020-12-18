'''
Adapted from https://github.com/rosinality/stylegan2-pytorch/blob/master/train.py
'''
import argparse
import os

import torch
from torch import nn, optim
from torch.utils import data
from torchvision import transforms, utils
from tqdm import tqdm

