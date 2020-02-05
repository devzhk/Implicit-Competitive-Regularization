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


def update_g(D, G, optimizer,
             batchsize, config, device):
    z = torch.randn((batchsize, config['z_dim']), device=device)
    d_fake = D(G(z))
    loss = get_loss(name=config['loss_type'], train_g=True,
                    d_fake=d_fake)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def update_d(real_x, D, G, optimizer,
             config, device, g_loss=False):
    z = torch.randn((real_x.shape[0], config['z_dim']), device=device)
    fake_x = G(z).detach()
    d_real = D(real_x)
    d_fake = D(fake_x)
    loss_d = get_loss(name=config['loss_type'], train_g=False,
                    d_real=d_real, d_fake=d_fake)
    if g_loss:
        loss_g = get_loss(name=config['loss_type'], train_g=True,
                          d_fake=d_fake)
    optimizer.zero_grad()
    loss_d.backward()
    optimizer.step()
    if g_loss:
        return loss_d, loss_g, d_real.mean().item(), d_fake.mean().item()
    else:
        return loss_d, d_real.mean().item(), d_fake.mean().item()


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

