import os
import time

import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from optims import SCGD, SCG
from train_utils import get_data, weights_init_d, weights_init_g, \
    get_diff, save_checkpoint, lr_scheduler, generate_data, icrScheduler, get_model
from losses import get_loss
from utils import cgd_trainer

# seed = torch.randint(0, 1000000, (1,))
seed = 2020
torch.manual_seed(seed=seed)
print('random seed : %d' % seed)


def train(config, tols, milestone, n=2, device='cpu'):
    """
    Train a model.

    Args:
        config: (todo): write your description
        tols: (float): write your description
        milestone: (list): write your description
        n: (array): write your description
        device: (todo): write your description
    """
    lr_d = config['lr_d']
    lr_g = config['lr_g']
    optim_type = config['optimizer']
    z_dim = config['z_dim']
    model_name = config['model']
    epoch_num = config['epoch_num']
    show_iter = config['show_iter']
    loss_name = config['loss_type']
    l2_penalty = config['d_penalty']
    logdir = config['logdir']
    start_n = config['startn']
    dataset = get_data(dataname=config['dataset'], path='../datas/%s' % config['datapath'])
    dataloader = DataLoader(dataset=dataset, batch_size=config['batchsize'],
                            shuffle=True, num_workers=4)
    D, G = get_model(model_name=model_name, z_dim=z_dim)
    D.apply(weights_init_d).to(device)
    G.apply(weights_init_g).to(device)
    if optim_type == 'SCGD':
        optimizer = SCGD(max_params=G.parameters(), min_params=D.parameters(),
                         lr_max=lr_g, lr_min=lr_d,
                         tol=tols['tol'], atol=tols['atol'],
                         device=device, solver='cg')
        scheduler = lr_scheduler(optimizer=optimizer, milestone=milestone)
    if config['checkpoint'] is not None:
        startPoint = config['checkpoint']
        chk = torch.load(startPoint)
        D.load_state_dict(chk['D'])
        G.load_state_dict(chk['G'])
        optimizer.load_state_dict(chk['optim'])
        print('Start from %s' % startPoint)
    gpu_num = config['gpu_num']
    if gpu_num > 1:
        D = nn.DataParallel(D, list(range(gpu_num)))
        G = nn.DataParallel(G, list(range(gpu_num)))
    timer = time.time()
    count = 0

    if model_name == 'DCGAN' or model_name == 'DCGAN-WBN':
        fixed_noise = torch.randn((64, z_dim, 1, 1), device=device)
    else:
        fixed_noise = torch.randn((64, z_dim), device=device)
    for e in range(epoch_num):
        scheduler.step(epoch=e)
        print('======Epoch: %d / %d======' % (e, epoch_num))
        for real_x in dataloader:
            optimizer.zero_grad()
            real_x = real_x[0].to(device)
            d_real = D(real_x)
            if model_name == 'DCGAN' or model_name == 'DCGAN-WBN':
                z = torch.randn((d_real.shape[0], z_dim, 1, 1), device=device)
            else:
                z = torch.randn((d_real.shape[0], z_dim), device=device)
            fake_x = G(z)
            d_fake = D(fake_x)
            loss = get_loss(name=loss_name, g_loss=False,
                            d_real=d_real, d_fake=d_fake,
                            l2_weight=l2_penalty, D=D)
            optimizer.step(loss)
            if (count + 1) % n == 0:
                optimizer.update(n)
            if count % show_iter == 0:
                time_cost = time.time() - timer
                print('Iter :%d , Loss: %.5f, time: %.3fs'
                      % (count, loss.item(), time_cost))
                timer = time.time()
                with torch.no_grad():
                    fake_img = G(fixed_noise).detach()
                    path = 'figs/%s_%s/' % (config['dataset'], logdir)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    vutils.save_image(fake_img, path + 'iter_%d.png' % (count + start_n), normalize=True)
                save_checkpoint(path=logdir,
                                name='%s-%s%.3f_%d.pth' % (optim_type, model_name, lr_g, count + start_n),
                                D=D, G=G, optimizer=optimizer)
            count += 1


def train_scg(config, tols, milestone, device='cpu'):
    """
    Train a model.

    Args:
        config: (todo): write your description
        tols: (float): write your description
        milestone: (str): write your description
        device: (todo): write your description
    """
    lr_d = config['lr_d']
    lr_g = config['lr_g']
    optim_type = config['optimizer']
    z_dim = config['z_dim']
    model_name = config['model']
    epoch_num = config['epoch_num']
    show_iter = config['show_iter']
    loss_name = config['loss_type']
    l2_penalty = config['d_penalty']
    logdir = config['logdir']
    start_n = config['startn']
    dataset = get_data(dataname=config['dataset'], path='../datas/%s' % config['datapath'])
    dataloader = DataLoader(dataset=dataset, batch_size=config['batchsize'],
                            shuffle=True, num_workers=4)
    inner_loader = DataLoader(dataset=dataset, batch_size=config['batchsize'],
                              shuffle=True, num_workers=4)
    D, G = get_model(model_name=model_name, z_dim=z_dim)
    D.apply(weights_init_d).to(device)
    G.apply(weights_init_g).to(device)
    optimizer = SCG(max_params=G.parameters(), min_params=D.parameters(),
                    lr_max=lr_g, lr_min=lr_d,
                    tol=tols['tol'], atol=tols['atol'],
                    dataloader=inner_loader,
                    device=device, solver='cg')
    scheduler = lr_scheduler(optimizer=optimizer, milestone=milestone)
    if config['checkpoint'] is not None:
        startPoint = config['checkpoint']
        chk = torch.load(startPoint)
        D.load_state_dict(chk['D'])
        G.load_state_dict(chk['G'])
        optimizer.load_state_dict(chk['optim'])
        print('Start from %s' % startPoint)
    gpu_num = config['gpu_num']
    if gpu_num > 1:
        D = nn.DataParallel(D, list(range(gpu_num)))
        G = nn.DataParallel(G, list(range(gpu_num)))

    timer = time.time()
    count = 0
    if model_name == 'DCGAN' or model_name == 'DCGAN-WBN':
        fixed_noise = torch.randn((64, z_dim, 1, 1), device=device)
    else:
        fixed_noise = torch.randn((64, z_dim), device=device)
    for e in range(epoch_num):
        scheduler.step(epoch=e)
        print('======Epoch: %d / %d======' % (e, epoch_num))
        for real_x in dataloader:
            optimizer.zero_grad()
            real_x = real_x[0]
            if model_name == 'DCGAN' or model_name == 'DCGAN-WBN':
                z = torch.randn((real_x.shape[0], z_dim, 1, 1), device=device)
            else:
                z = torch.randn((real_x.shape[0], z_dim), device=device)
            def closure(train_x):
                """
                : parametermine the loss : return :

                Args:
                    train_x: (int): write your description
                """
                train_x = train_x.to(device)
                fake_x = G(z)
                d_fake = D(fake_x)
                d_real = D(train_x)
                loss = get_loss(name=loss_name, g_loss=False,
                                d_real=d_real, d_fake=d_fake,
                                l2_weight=l2_penalty, D=D)
                return loss
            loss = optimizer.step(closure=closure, img=real_x)
            if count % show_iter == 0:
                time_cost = time.time() - timer
                print('Iter :%d , Loss: %.5f, time: %.3fs'
                      % (count, loss.item(), time_cost))
                timer = time.time()
                with torch.no_grad():
                    fake_img = G(fixed_noise).detach()
                    path = 'figs/%s_%s/' % (config['dataset'], logdir)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    vutils.save_image(fake_img, path + 'iter_%d.png' % (count + start_n), normalize=True)
                save_checkpoint(path=logdir,
                                name='%s-%s%.3f_%d.pth' % (optim_type, model_name, lr_g, count + start_n),
                                D=D, G=G, optimizer=optimizer)
            count += 1


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    parser = cgd_trainer()
    parser.add_argument('--strategy', type=str, default='scg')
    config = vars(parser.parse_args())
    print(config)
    lr_g = config['lr_g']
    lr_d = config['lr_d']
    milestones = {'0': (lr_g, lr_d)}
    tols = {'tol': config['tol'], 'atol': config['atol']}
    if config['strategy'] == 'scg':
        train_scg(config=config, tols=tols, milestone=milestones, device=device)
    else:
        train(config=config, tols=tols, milestone=milestones, n=3, device=device)
