import os
import csv
import time

import torch
import torch.nn as nn
import torchvision.utils as vutils

from torch.optim.sgd import SGD
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from GANs import dc_G, dc_D, \
    GoodGenerator, GoodDiscriminator, \
    DC_generator, DC_discriminator
from optims.cgd import BCGD
from train_utils import get_data, weights_init_d, weights_init_g, \
    get_diff, save_checkpoint, lr_scheduler, generate_data
from losses import get_loss


def train_mnist(epoch_num=10, show_iter=100, logdir='test',
                model_weight=None, load_d=False, load_g=False,
                compare_path=None, info_time=100, run_select=None,
                device='cpu'):
    lr_d = 0.01
    lr_g = 0.01
    batchsize = 128
    z_dim = 96
    print('MNIST, discriminator lr: %.3f, generator lr: %.3f' %(lr_d, lr_g))
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
    g_optimizer = SGD(G.parameters(), lr=lr_g)
    timer = time.time()
    count = 0
    fixed_noise = torch.randn((64, z_dim), device=device)
    for e in range(epoch_num):
        for real_x in dataloader:
            real_x = real_x[0].to(device)
            d_real = D(real_x)
            z = torch.randn((d_real.shape[0], z_dim), device=device)
            fake_x = G(z)
            fake_x_c = fake_x.clone().detach()
            # update generator
            d_fake = D(fake_x)

            writer.add_scalars('Discriminator output', {'Generated image': d_fake.mean().item(),
                                                        'Real image': d_real.mean().item()},
                               global_step=count)
            G_loss = get_loss(name='JSD', g_loss=True, d_fake=d_fake)
            g_optimizer.zero_grad()
            G_loss.backward()
            g_optimizer.step()
            gg = torch.norm(
                torch.cat([p.grad.contiguous().view(-1) for p in G.parameters()]), p=2)

            d_fake_c = D(fake_x_c)
            D_loss = get_loss(name='JSD', g_loss=False, d_real=d_real, d_fake=d_fake_c)
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

            gd = torch.norm(
                torch.cat([p.grad.contiguous().view(-1) for p in D.parameters()]), p=2)

            writer.add_scalars('Loss', {'D_loss': D_loss.item(),
                                        'G_loss': G_loss.item()}, global_step=count)
            writer.add_scalars('Grad', {'D grad': gd.item(),
                                        'G grad': gg.item()}, global_step=count)
            if count % show_iter == 0:
                time_cost = time.time() - timer
                print('Iter :%d , D_loss: %.5f, G_loss: %.5f, time: %.3fs' % (
                    count, D_loss.item(), G_loss.item(), time_cost))
                timer = time.time()
                with torch.no_grad():
                    fake_img = G(fixed_noise).detach()
                    path = 'figs/%s/' % logdir
                    if not os.path.exists(path):
                        os.makedirs(path)
                    vutils.save_image(fake_img, path + 'iter_%d.png' % count, normalize=True)
                save_checkpoint(path=logdir,
                                name='SGD-%.3f_%d.pth' % (lr_d, count),
                                D=D, G=G)
            count += 1
    writer.close()


def train_cgd(epoch_num=10, milestone=None,
              loss_name='WGAN', model_name='dc', data_path='None',
              show_iter=100, logdir='test', dataname='cifar10',
              device='cpu', gpu_num=1, collect_info=False):
    lr_d = 0.001
    lr_g = 0.001
    batchsize = 64
    z_dim = 100
    dataset = get_data(dataname=dataname, path='../datas/%s' % data_path)
    dataloader = DataLoader(dataset=dataset, batch_size=batchsize, shuffle=True,
                            num_workers=4)
    if model_name == 'dc':
        D = GoodDiscriminator().to(device)
        G = GoodGenerator().to(device)
    elif model_name == 'DCGAN':
        D = DC_discriminator().to(device)
        G = DC_generator(z_dim=z_dim).to(device)
    if gpu_num > 1:
        D = nn.DataParallel(D, list(range(gpu_num)))
        G = nn.DataParallel(G, list(range(gpu_num)))
    D.apply(weights_init_d)
    G.apply(weights_init_g)
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    writer = SummaryWriter(log_dir='logs/%s/%s_%.3f' % (logdir, current_time, lr_d))
    optimizer = BCGD(max_params=G.parameters(), min_params=D.parameters(),
                     lr_max=lr_g, lr_min=lr_d, device=device)
    scheduler = lr_scheduler(optimizer=optimizer, milestone=milestone)
    timer = time.time()
    count = 0
    if model_name == 'dc':
        fixed_noise = torch.randn((64, z_dim), device=device)
    else:
        fixed_noise = torch.randn((64, z_dim, 1, 1), device=device)
    for e in range(epoch_num):
        scheduler.step(epoch=e)
        print('======Epoch: %d / %d======' % (e, epoch_num))
        for real_x in dataloader:
            real_x = real_x[0].to(device)
            d_real = D(real_x)
            if model_name == 'dc':
                z = torch.randn((d_real.shape[0], z_dim), device=device)
            else:
                z = torch.randn((d_real.shape[0], z_dim, 1, 1), device=device)
            fake_x = G(z)
            d_fake = D(fake_x)
            loss = get_loss(name=loss_name, g_loss=False, d_real=d_real, d_fake=d_fake)
            optimizer.zero_grad()
            optimizer.step(loss)

            if count % show_iter == 0:
                time_cost = time.time() - timer
                print('Iter :%d , Loss: %.5f, time: %.3fs'
                      % (count, loss.item(), time_cost))
                timer = time.time()
                with torch.no_grad():
                    fake_img = G(fixed_noise).detach()
                    path = 'figs/%s_%s/' % (dataname, logdir)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    vutils.save_image(fake_img, path + 'iter_%d.png' % count, normalize=True)
                save_checkpoint(path=logdir,
                                name='CGD-%.3f%.3f_%d.pth' % (lr_d, lr_g, count),
                                D=D, G=G)
            writer.add_scalars('Discriminator output', {'Generated image': d_fake.mean().item(),
                                                        'Real image': d_real.mean().item()},
                               global_step=count)
            writer.add_scalar('Loss', loss.item(), global_step=count)
            if collect_info:
                cgd_info = optimizer.getinfo()
                writer.add_scalar('Conjugate Gradient/iter num', cgd_info['iter_num'], global_step=count)
                writer.add_scalar('Conjugate Gradient/running time', cgd_info['time'], global_step=count)
                writer.add_scalars('Delta', {'D gradient': cgd_info['grad_y'],
                                             'G gradient': cgd_info['grad_x'],
                                             'D hvp': cgd_info['hvp_y'],
                                             'G hvp': cgd_info['hvp_x'],
                                             'D cg': cgd_info['cg_y'],
                                             'G cg': cgd_info['cg_x']},
                                   global_step=count)
            count += 1
    writer.close()


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # chk_path = 'checkpoints/0.00000MNIST-0.0100/SGD-0.01000_9000.pth'
    # generate_data(model_weight=chk_path, path='figs/select/Fixed_1000.pt', device=device)
    # train_mnist(epoch_num=30, show_iter=500, logdir='select',
    #             model_weight=chk_path, load_d=True, load_g=True,
    #             compare_path=chk_path, info_time=100, run_select='figs/select/Fixed_1000.pt',
    #             device=device)

    # train_cgd(epoch_num=40, milestone=(25, 30, 35), show_iter=500, logdir='cifar',
    #           dataname='CIFAR10', loss_name='WGAN', model_name='dc',
    #           device=device, gpu_num=2, collect_info=True)
    milestones = {'1': (0.01, 0.01),
                  '3': (0.001, 0.001),
                  '4': (0.0001, 0.0001)}
    train_cgd(epoch_num=5, milestone=milestones,
              show_iter=500, logdir='bedroom_test',
              data_path='lsun', dataname='LSUN-bedroom',
              loss_name='WGAN', model_name='DCGAN',
              device=device, gpu_num=2, collect_info=True)