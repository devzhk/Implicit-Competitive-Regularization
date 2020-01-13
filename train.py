import csv
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.optim.rmsprop import RMSprop
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST

from GANs.models import dc_D, dc_G, dc_d, dc_g, GoodDiscriminator, GoodGenerator, GoodDiscriminatord
from CGDs.optimizers import BCGD
from CGDs.cgd_utils import zero_grad
from utils import prepare_parser

seed = torch.randint(0, 1000000, (1,))
torch.manual_seed(seed=seed)
print('random seed : %d' % seed)


def detransform(x):
    return (x + 1.0) / 2.0


def weights_init_d(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.005)


def weights_init_g(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.005)


def save_checkpoint(path, name, optimizer, D, G):
    chk_name = 'checkpoints/' + path
    if not os.path.exists(chk_name):
        os.makedirs(chk_name)
    d_state_dict = D.state_dict()
    g_state_dict = G.state_dict()
    optim_dict = optimizer.state_dict()
    torch.save({
        'D': d_state_dict,
        'G': g_state_dict,
        'optim': optim_dict
    }, chk_name + name)
    print('model is saved at %s' % chk_name + name)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    parser = prepare_parser()
    config = vars(parser.parse_args())
    print(config)

    if config['dataset'] == 'CIFAR10':
        dataset = CIFAR10(config['datapath'], train=True,
                          transform=transforms.Compose([transforms.ToTensor(),
                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                        ]),
                          download=True)
        D = GoodDiscriminator().to(device)
        G = GoodGenerator().to(device)
        fixed_noise = torch.randn((64, config['z_dim']), device=device)
    else:
        dataset = MNIST(config['datapath'], train=True,
                        transform=transforms.Compose([transforms.ToTensor(),
                                                      transforms.Normalize(0.5, 0.5)
                                                      ]),
                        download=True)
        D = dc_D()
        G = dc_G(z_dim=config['z_dim'])


    if config['gpu_num'] > 1:
        D = nn.DataParallel(D, list(range(config['gpu_num'])))
        G = nn.DataParallel(G, list(range(config['gpu_num'])))
    D.apply(weights_init_d)
    G.apply(weights_init_g)
    dataloader = DataLoader(dataset=dataset, batch_size=config['batchsize'], shuffle=True, num_workers=4)
    writer = SummaryWriter(log_dir='logs/' + config['logdir'])
    # f = open(config['logdir'] + '/scores.csv', 'w')
    # score_writer = csv.DictWriter(f, ['iter_num', 'is_mean', 'is_std'])
    iter_num = 0
    criterion = nn.BCEWithLogitsLoss()

    if config['optimizer'] == 'BCGD':
        optimizer = BCGD(max_params=G.parameters(), min_params=D.parameters(),
                         lr=config['lr_d'], momentum=config['momentum'],
                         device=device, collect_info=config['collect_info'])

    for e in range(config['epoch_num']):
        for real_x in dataloader:
            real_x = real_x[0].to(device)
            d_real = D(real_x)
            z = torch.randn((real_x.shape[0], config['z_dim']), device=device)
            fake_x = G(z)
            d_fake = D(fake_x)
            if config['loss_type'] == 'WGAN':
                loss = d_fake.mean() - d_real.mean()
            elif config['loss_type'] == 'JSD':
                loss = criterion(d_real, torch.ones(d_real.shape, device=device)) + \
                       criterion(d_fake, torch.zeros(d_fake.shape, device=device))
            else:
                raise ValueError('invalid loss type')
            optimizer.zero_grad()
            optimizer.step(loss=loss)

            writer.add_scalar('loss', loss.item(), iter_num)
            writer.add_scalars('disc output', {'real': d_real.mean().item(),
                                               'fake': d_fake.mean().item()},
                               iter_num)
            dn = torch.norm(torch.cat([p.contiguous().view(-1) for p in D.parameters()]), p=2).detach_()
            gn = torch.norm(torch.cat([p.contiguous().view(-1) for p in G.parameters()]), p=2).detach_()
            writer.add_scalars('Weight norm', {'generator': gn, 'discriminator': dn}, iter_num)

            if iter_num % config['show_iter'] == 0:
                print('Iter_num : {}, loss: {}'.format(iter_num, loss.item()))
                fake_image = G(z).detach()
                images = detransform(fake_image)
                writer.add_images('Generated images', images, global_step=iter_num, dataformats='NCHW')
                save_checkpoint(path=config['logdir'],
                                name='/%s_%d.pth' % (config['optimizer'], iter_num),
                                D=D, G=G, optimizer=optimizer)
            iter_num += 1
    writer.close()
