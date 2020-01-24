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

from CGDs.optimizers import BCGD
from utils import train_seq_parser
from train_utils import data_model, save_checkpoint, detransform, \
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
    parser = train_seq_parser()
    config = vars(parser.parse_args())
    print(config)
    mode = config['optimizer']
    momentum = config['momentum']
    d_iter = config['d_iter']
    model_weight = config['model_weight']
    dataset, D, G = data_model(config)
    D = D.to(device)
    G = G.to(device)
    if mode == 'Adam':
        optimizer_d = Adam(D.parameters(), lr=config['lr_d'], betas=(momentum, 0.999))
        optimizer_g = Adam(G.parameters(), lr=config['lr_g'], betas=(momentum, 0.999))
    elif mode == 'SGD':
        optimizer_d = SGD(D.parameters(), lr=config['lr_d'], momentum=momentum)
        optimizer_g = SGD(G.parameters(), lr=config['lr_g'], momentum=momentum)
    if model_weight == 'random':
        D.apply(weights_init_d)
        G.apply(weights_init_g)

    if config['gpu_num'] > 1:
        D = nn.DataParallel(D, list(range(config['gpu_num'])))
        G = nn.DataParallel(G, list(range(config['gpu_num'])))
    fixed_noise = torch.randn((64, config['z_dim']), device=device)

    dataloader = DataLoader(dataset=dataset, batch_size=config['batchsize'], shuffle=True, num_workers=4)
    writer = SummaryWriter(log_dir='logs/' + config['logdir'])
    # f = open(config['logdir'] + '/scores.csv', 'w')
    # score_writer = csv.DictWriter(f, ['iter_num', 'is_mean', 'is_std'])
    iter_num = 0

    for e in range(config['epoch_num']):
        for real_x in dataloader:
            iter_num += 1
            real_x = real_x[0].to(device)
            if d_iter == 0:
                loss_d, loss_g, d_real, d_fake = \
                    update_d(real_x, D=D, G=G,
                             optimizer=optimizer_d,
                             config=config, device=device, g_loss=True)
            elif d_iter >= 1:
                loss_d, d_real, d_fake = update_d(real_x, D=D, G=G,
                                                  optimizer=optimizer_d,
                                                  config=config, device=device)
                if iter_num % d_iter == 0:
                    loss_g = update_g(D=D, G=G, optimizer=optimizer_g, config=config, device=device)
            else:
                raise ValueError('invalid d_iter (int, >=0)')




            writer.add_scalar('loss', loss.item(), iter_num)
            writer.add_scalars('disc output', {'real': d_real.mean().item(),
                                               'fake': d_fake.mean().item()},
                               iter_num)
            dn = torch.norm(torch.cat([p.contiguous().view(-1) for p in D.parameters()]), p=2).detach_()
            gn = torch.norm(torch.cat([p.contiguous().view(-1) for p in G.parameters()]), p=2).detach_()
            writer.add_scalars('Weight norm', {'generator': gn, 'discriminator': dn}, iter_num)

            if iter_num % config['show_iter'] == 0:
                print('Iter_num : {}, loss: {}'.format(iter_num, loss.item()))
                fake_image = G(fixed_noise).detach()
                images = detransform(fake_image)
                writer.add_images('Generated images', images, global_step=iter_num, dataformats='NCHW')
                save_checkpoint(path=config['logdir'],
                                name='/%s_%d.pth' % (config['optimizer'], iter_num),
                                D=D, G=G, optimizer=optimizer)
            iter_num += 1
    writer.close()
