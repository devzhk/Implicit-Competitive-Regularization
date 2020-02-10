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

from GANs import dc_G, dc_D
from utils import train_seq_parser
from train_utils import get_data, save_checkpoint, detransform, \
    weights_init_d, weights_init_g, get_diff
from losses import get_loss


seed = torch.randint(0, 1000000, (1,))
torch.manual_seed(seed=seed)
print('random seed : %d' % seed)


def train_d(epoch_num=10, logdir='test',
            loss_name='JSD', show_iter=500,
            model_weight=None, load_d=False, load_g=False,
            compare_path=None, info_time=100, run_select=None,
            device='cpu'):
    lr_d = 0.01
    lr_g = 0.01
    batchsize = 128
    z_dim = 96
    print('MNIST, discriminator lr: %.3f' % lr_d)
    dataset = get_data(dataname='MNIST', path='../datas/mnist')
    dataloader = DataLoader(dataset=dataset, batch_size=batchsize, shuffle=True,
                            num_workers=4)
    D = dc_D().to(device)
    G = dc_G(z_dim=z_dim).to(device)
    D.apply(weights_init_d)
    G.apply(weights_init_g)
    if model_weight is not None:
        chk = torch.load(model_weight)
        if load_d:
            D.load_state_dict(chk['D'])
            print('Load D from %s' % model_weight)
        if load_g:
            G.load_state_dict(chk['G'])
            print('Load G from %s' % model_weight)
    if compare_path is not None:
        discriminator = dc_D().to(device)
        model_weight = torch.load(compare_path)
        discriminator.load_state_dict(model_weight['D'])
        model_vec = torch.cat([p.contiguous().view(-1) for p in discriminator.parameters()])
        print('Load discriminator from %s' % compare_path)
    if run_select is not None:
        fixed_data = torch.load(run_select)
        real_set = fixed_data['real_set']
        fake_set = fixed_data['fake_set']
        real_d = fixed_data['real_d']
        fake_d = fixed_data['fake_d']
        fixed_vec = fixed_data['pred_vec']
        print('load fixed data set')
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    writer = SummaryWriter(log_dir='logs/%s/%s_%.3f' % (logdir, current_time, lr_d))
    d_optimizer = SGD(D.parameters(), lr=lr_d)
    timer = time.time()
    count = 0
    d_losses = []
    g_losses = []
    for e in range(epoch_num):
        tol_correct = 0
        tol_dloss = 0
        tol_gloss = 0
        for real_x in dataloader:
            real_x = real_x[0].to(device)
            d_real = D(real_x)
            z = torch.randn((real_x.shape[0], z_dim), device=device)
            fake_x = G(z)
            d_fake = D(fake_x)
            D_loss = get_loss(name=loss_name, g_loss=False, d_real=d_real, d_fake=d_fake)
            tol_dloss += D_loss.item() * real_x.shape[0]
            G_loss = get_loss(name=loss_name, g_loss=True, d_real=d_real, d_fake=d_fake)
            tol_gloss += G_loss.item() * fake_x.shape[0]
            if compare_path is not None and count % info_time == 0:
                diff = get_diff(net=D, model_vec=model_vec)
                writer.add_scalar('Distance from checkpoint', diff.item(), global_step=count)
                if run_select is not None:
                    with torch.no_grad():
                        d_real_set = D(real_set)
                        d_fake_set = D(fake_set)
                        diff_real = torch.norm(d_real_set - real_d, p=2)
                        diff_fake = torch.norm(d_fake_set - fake_d, p=2)
                        d_vec = torch.cat([d_real_set, d_fake_set])
                        diff = torch.norm(d_vec.sub_(fixed_vec), p=2)
                        writer.add_scalars('L2 norm of pred difference',
                                           {'Total': diff.item(),
                                            'real set': diff_real.item(),
                                            'fake set': diff_fake.item()},
                                           global_step=count)
            d_optimizer.zero_grad()
            D_loss.backward()
            d_optimizer.step()
            tol_correct += (d_real > 0).sum().item() + (d_fake < 0).sum().item()

            gd = torch.norm(
                torch.cat([p.grad.contiguous().view(-1) for p in D.parameters()]), p=2)
            gg = torch.norm(
                torch.cat([p.grad.contiguous().view(-1) for p in G.parameters()]), p=2)

            writer.add_scalars('Loss', {'D_loss': D_loss.item(),
                                        'G_loss': G_loss.item()}, global_step=count)
            writer.add_scalars('Grad', {'D grad': gd.item(),
                                        'G grad': gg.item()}, global_step=count)
            writer.add_scalars('Discriminator output', {'Generated image': d_fake.mean().item(),
                                                        'Real image': d_real.mean().item()},
                               global_step=count)
            if count % show_iter == 0:
                time_cost = time.time() - timer
                print('Iter :%d , D_loss: %.5f, G_loss: %.5f, time: %.3fs' % (
                    count, D_loss.item(), G_loss.item(), time_cost))
                timer = time.time()
                save_checkpoint(path=logdir,
                                name='FixG-%.3f_%d.pth' % (lr_d, count),
                                D=D, G=G)
            count += 1
    writer.close()


def train_g(epoch_num=10, logdir='test',
            loss_name='JSD', show_iter=500,
            model_weight=None, load_d=False, load_g=False,
            device='cpu'):
    lr_d = 0.01
    lr_g = 0.01
    batchsize = 128
    z_dim = 96
    print('MNIST, discriminator lr: %.3f' % lr_d)
    dataset = get_data(dataname='MNIST', path='../datas/mnist')
    dataloader = DataLoader(dataset=dataset, batch_size=batchsize, shuffle=True,
                            num_workers=4)
    D = dc_D().to(device)
    G = dc_G(z_dim=z_dim).to(device)
    D.apply(weights_init_d)
    G.apply(weights_init_g)
    if model_weight is not None:
        chk = torch.load(model_weight)
        if load_d:
            D.load_state_dict(chk['D'])
            print('Load D from %s' % model_weight)
        if load_g:
            G.load_state_dict(chk['G'])
            print('Load G from %s' % model_weight)
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    writer = SummaryWriter(log_dir='logs/%s/%s_%.3f' % (logdir, current_time, lr_g))
    d_optimizer = SGD(D.parameters(), lr=lr_d)
    g_optimizer = SGD(G.parameters(), lr=lr_g)
    timer = time.time()
    count = 0
    for e in range(epoch_num):
        for real_x in dataloader:
            real_x = real_x[0].to(device)
            d_real = D(real_x)
            z = torch.randn((real_x.shape[0], z_dim), device=device)
            fake_x = G(z)
            d_fake = D(fake_x)
            D_loss = get_loss(name=loss_name, g_loss=False, d_real=d_real, d_fake=d_fake)
            G_loss = get_loss(name=loss_name, g_loss=True, d_real=d_real, d_fake=d_fake)
            d_optimizer.zero_grad()
            g_optimizer.zero_grad()
            G_loss.backward()
            g_optimizer.step()
            print('D_loss: {}, G_loss: {}'.format(D_loss.item(), G_loss.item()))
            writer.add_scalars('Loss', {'D_loss': D_loss.item(),
                                        'G_loss': G_loss.item()},
                               global_step=count)
            writer.add_scalars('Discriminator output', {'Generated image': d_fake.mean().item(),
                                                        'Real image': d_real.mean().item()},
                               global_step=count)
            if count % show_iter == 0:
                time_cost = time.time() - timer
                print('Iter :%d , D_loss: %.5f, G_loss: %.5f, time: %.3fs' % (
                    count, D_loss.item(), G_loss.item(), time_cost))
                timer = time.time()
                save_checkpoint(path=logdir,
                                name='FixD-%.3f_%d.pth' % (lr_d, count),
                                D=D, G=G)
            count += 1
        writer.close()


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    chk_path = 'checkpoints/0.00000MNIST-0.0100/SGD-0.01000_9000.pth'
    # train_d(epoch_num=30, show_iter=500,
    #         logdir='select', loss_name='JSD',
    #         model_weight=chk_path, load_d=True, load_g=True,
    #         compare_path=chk_path, info_time=100, run_select='figs/select/Fixed_1000.pt',
    #         device=device)
    # train_d(epoch_num=30, show_iter=500,
    #         logdir='sgd', loss_name='JSD',
    #         model_weight=chk_path, load_d=True, load_g=True,
    #         compare_path=None, info_time=100,
    #         device=device)

    fixG_path = 'checkpoints/sgd/FixG-0.010_14000.pth'
    train_g(epoch_num=30, show_iter=500,
            logdir='sgd_new', loss_name='JSD',
            model_weight=fixG_path, load_d=True, load_g=True,
            device=device)
