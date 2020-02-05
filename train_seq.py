import csv
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
import torchvision.utils as vutils

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from utils import train_seq_parser
from train_utils import get_model, get_data, save_checkpoint, detransform, \
    weights_init_d, weights_init_g
from losses import get_loss


seed = torch.randint(0, 1000000, (1,))
torch.manual_seed(seed=seed)
print('random seed : %d' % seed)



if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

