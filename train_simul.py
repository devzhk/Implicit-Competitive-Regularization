import os
import time
import yaml

import torch
import torch.nn as nn
import torchvision.utils as vutils

from torch.optim.sgd import SGD
from torch.optim.rmsprop import RMSprop

from torch.utils.data import DataLoader

from GANs import dc_G, dc_D
from optims import ACGD, BCGD, CNAG
from train_utils import get_data, weights_init_d, weights_init_g, \
    get_diff, save_checkpoint, lr_scheduler, generate_data, icrScheduler, get_model
from losses import get_loss
from utils import cgd_trainer

try:
    import wandb

except ImportError:
    wandb = None

# seed = torch.randint(0, 1000000, (1,))
seed = 2020
torch.manual_seed(seed=seed)
print('random seed : %d' % seed)


def train_sim(epoch_num=10, optim_type='ACGD',
              startPoint=None, start_n=0,
              z_dim=128, batchsize=64,
              l2_penalty=0.0, momentum=0.0, log=False,
              loss_name='WGAN', model_name='dc',
              model_config=None,
              data_path='None',
              show_iter=100, logdir='test',
              dataname='CIFAR10',
              device='cpu', gpu_num=1):
    lr_d = 1e-4
    lr_g = 1e-4
    dataset = get_data(dataname=dataname, path=data_path)
    dataloader = DataLoader(dataset=dataset, batch_size=batchsize, shuffle=True,
                            num_workers=4)
    D, G = get_model(model_name=model_name, z_dim=z_dim, configs=model_config)
    D.apply(weights_init_d).to(device)
    G.apply(weights_init_g).to(device)

    optim_d = RMSprop(D.parameters(), lr=lr_d)
    optim_g = RMSprop(G.parameters(), lr=lr_g)

    if startPoint is not None:
        chk = torch.load(startPoint)
        D.load_state_dict(chk['D'])
        G.load_state_dict(chk['G'])
        optim_d.load_state_dict(chk['d_optim'])
        optim_g.load_state_dict(chk['g_optim'])
        print('Start from %s' % startPoint)
    if gpu_num > 1:
        D = nn.DataParallel(D, list(range(gpu_num)))
        G = nn.DataParallel(G, list(range(gpu_num)))
    timer = time.time()
    count = 0
    if 'DCGAN' in model_name:
        fixed_noise = torch.randn((64, z_dim, 1, 1), device=device)
    else:
        fixed_noise = torch.randn((64, z_dim), device=device)
    for e in range(epoch_num):
        print('======Epoch: %d / %d======' % (e, epoch_num))
        for real_x in dataloader:
            real_x = real_x[0].to(device)
            d_real = D(real_x)
            if 'DCGAN' in model_name:
                z = torch.randn((d_real.shape[0], z_dim, 1, 1), device=device)
            else:
                z = torch.randn((d_real.shape[0], z_dim), device=device)
            fake_x = G(z)
            d_fake = D(fake_x)
            loss = get_loss(name=loss_name, g_loss=False,
                            d_real=d_real, d_fake=d_fake,
                            l2_weight=l2_penalty, D=D)
            D.zero_grad()
            G.zero_grad()
            loss.backward()
            optim_d.step()
            optim_g.step()

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
                    vutils.save_image(fake_img, path + 'iter_%d.png' % (count + start_n), normalize=True)
                save_checkpoint(path=logdir,
                                name='%s-%s%.3f_%d.pth' % (optim_type, model_name, lr_g, count + start_n),
                                D=D, G=G, optimizer=optim_d, g_optimizer=optim_g)
            if wandb and log:
                wandb.log(
                    {
                        'Real score': d_real.mean().item(),
                        'Fake score': d_fake.mean().item(),
                        'Loss': loss.item()
                    }
                )
            count += 1


def train_mnist(epoch_num=10, show_iter=100, logdir='test',
                model_weight=None, load_d=False, load_g=False,
                compare_path=None, info_time=100, run_select=None,
                dataname='CIFAR10', data_path='None',
                device='cpu'):
    lr_d = 0.01
    lr_g = 0.01
    batchsize = 128
    z_dim = 96
    print('MNIST, discriminator lr: %.3f, generator lr: %.3f' % (lr_d, lr_g))
    dataset = get_data(dataname=dataname, path=data_path)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batchsize,
                            shuffle=True,
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

            # writer.add_scalars('Discriminator output', {'Generated image': d_fake.mean().item(),
            #                                             'Real image': d_real.mean().item()},
            #                    global_step=count)
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
                # writer.add_scalar('Distance from checkpoint', diff.item(), global_step=count)
                if run_select is not None:
                    with torch.no_grad():
                        d_real_set = D(real_set)
                        d_fake_set = D(fake_set)
                        diff_real = torch.norm(d_real_set - real_d, p=2)
                        diff_fake = torch.norm(d_fake_set - fake_d, p=2)
                        d_vec = torch.cat([d_real_set, d_fake_set])
                        diff = torch.norm(d_vec.sub_(fixed_vec), p=2)
                        # writer.add_scalars('L2 norm of pred difference',
                        #                    {'Total': diff.item(),
                        #                     'real set': diff_real.item(),
                        #                     'fake set': diff_fake.item()},
                        #                    global_step=count)
            d_optimizer.zero_grad()
            D_loss.backward()
            d_optimizer.step()
            gd = torch.norm(
                torch.cat([p.grad.contiguous().view(-1) for p in D.parameters()]), p=2)
            # writer.add_scalars('Loss', {'D_loss': D_loss.item(),
            #                             'G_loss': G_loss.item()}, global_step=count)
            # writer.add_scalars('Grad', {'D grad': gd.item(),
            #                             'G grad': gg.item()}, global_step=count)
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


def train_cgd(epoch_num=10, optim_type='ACGD',
              startPoint=None, start_n=0,
              z_dim=128, batchsize=64,
              tols={'tol': 1e-10, 'atol': 1e-16},
              l2_penalty=0.0, momentum=0.0,
              loss_name='WGAN', model_name='dc',
              model_config=None,
              data_path='None',
              show_iter=100, logdir='test',
              dataname='CIFAR10',
              device='cpu', gpu_num=1,
              ada_train=True, log=False,
              collect_info=False, args=None):
    lr_d = args['lr_d']
    lr_g = args['lr_g']
    dataset = get_data(dataname=dataname, path=data_path)
    dataloader = DataLoader(dataset=dataset, batch_size=batchsize, shuffle=True,
                            num_workers=4)
    D, G = get_model(model_name=model_name, z_dim=z_dim, configs=model_config)
    D.apply(weights_init_d).to(device)
    G.apply(weights_init_g).to(device)
    if optim_type == 'BCGD':
        optimizer = BCGD(max_params=G.parameters(), min_params=D.parameters(),
                         lr_max=lr_g, lr_min=lr_d, momentum=momentum,
                         tol=tols['tol'], atol=tols['atol'],
                         device=device)
        # scheduler = lr_scheduler(optimizer=optimizer, milestone=milestone)
    elif optim_type == 'CNAG':
        optimizer = CNAG(max_params=G.parameters(), min_params=D.parameters(),
                        lr=lr_d, alpha=1.0, device=device)
        # scheduler = icrScheduler(optimizer, milestone)
    elif optim_type == 'ACGD':
        optimizer = ACGD(max_params=G.parameters(), min_params=D.parameters(),
                         lr_max=lr_g, lr_min=lr_d,
                         tol=tols['tol'], atol=tols['atol'],
                         device=device, solver='cg')
        # scheduler = lr_scheduler(optimizer=optimizer, milestone=milestone)
    if startPoint is not None:
        chk = torch.load(startPoint)
        D.load_state_dict(chk['D'])
        G.load_state_dict(chk['G'])
        # optimizer.load_state_dict(chk['optim'])
        print('Start from %s' % startPoint)
    if gpu_num > 1:
        D = nn.DataParallel(D, list(range(gpu_num)))
        G = nn.DataParallel(G, list(range(gpu_num)))
    timer = time.time()
    count = 0
    if 'DCGAN' in model_name:
        fixed_noise = torch.randn((64, z_dim, 1, 1), device=device)
    else:
        fixed_noise = torch.randn((64, z_dim), device=device)

    mod = 10
    accs = torch.tensor([0.8 for _ in range(mod)])

    for e in range(epoch_num):
        # scheduler.step(epoch=e)
        print('======Epoch: %d / %d======' % (e, epoch_num))
        for real_x in dataloader:
            real_x = real_x[0].to(device)
            d_real = D(real_x)
            if 'DCGAN' in model_name:
                z = torch.randn((d_real.shape[0], z_dim, 1, 1), device=device)
            else:
                z = torch.randn((d_real.shape[0], z_dim), device=device)
            fake_x = G(z)
            d_fake = D(fake_x)
            loss = get_loss(name=loss_name, g_loss=False,
                            d_real=d_real, d_fake=d_fake,
                            l2_weight=l2_penalty, D=D)
            optimizer.zero_grad()
            optimizer.step(loss)

            num_correct = torch.sum(d_real > 0) + torch.sum(d_fake < 0)
            acc = num_correct.item() / (d_real.shape[0] + d_fake.shape[0])
            accs[count % mod] = acc
            acc_indicator = sum(accs) / mod
            if acc_indicator > 0.9:
                ada_ratio = 0.05
            elif acc_indicator < 0.80:
                ada_ratio = 0.1
            else:
                ada_ratio = 1.0
            if ada_train:
                optimizer.set_lr(lr_max=lr_g, lr_min=ada_ratio * lr_d)

            if count % show_iter == 0 and count != 0:
                time_cost = time.time() - timer
                print('Iter :%d , Loss: %.5f, time: %.3fs'
                      % (count, loss.item(), time_cost))
                timer = time.time()
                with torch.no_grad():
                    fake_img = G(fixed_noise).detach()
                    path = 'figs/%s_%s/' % (dataname, logdir)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    vutils.save_image(fake_img, path + 'iter_%d.png' % (count + start_n), normalize=True)
                save_checkpoint(path=logdir,
                                name='%s-%s_%d.pth' % (optim_type, model_name, count + start_n),
                                D=D, G=G, optimizer=optimizer)
            if wandb and log:
                wandb.log(
                    {
                        'Real score': d_real.mean().item(),
                        'Fake score': d_fake.mean().item(),
                        'Loss': loss.item(),
                        'Acc_indicator': acc_indicator,
                        'Ada ratio': ada_ratio
                    },
                    step=count,
                )

            if collect_info and wandb:
                cgd_info = optimizer.get_info()
                wandb.log(
                    {
                        'CG iter num': cgd_info['iter_num'],
                        'CG runtime': cgd_info['time'],
                        'D gradient': cgd_info['grad_y'],
                        'G gradient': cgd_info['grad_x'],
                        'D hvp': cgd_info['hvp_y'],
                        'G hvp': cgd_info['hvp_x'],
                        'D cg': cgd_info['cg_y'],
                        'G cg': cgd_info['cg_x']
                    },
                    step=count
                )
            count += 1


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    parser = cgd_trainer()
    config = vars(parser.parse_args())
    print(config)
    # load model configuration parameters if any
    model_args = None
    if config['model_config'] is not None:
        with open(config['model_config'], 'r') as configFile:
            model_config = yaml.safe_load(configFile)
        model_args = model_config['parameters']

    start_n = config['startn']
    chk_path = config['checkpoint']
    lr_g = config['lr_g']
    lr_d = config['lr_d']
    # milestones = {'0': (lr_g, lr_d)}
    tols = {'tol': config['tol'], 'atol': config['atol']}
    print(tols)

    if wandb and config['log']:
        wandb.init(project="%s-ada-acgd" % config['dataset'],
                   config={'lr_g': lr_g,
                           'lr_d': lr_d,
                           'Model name': config['model'],
                           'Batchsize': config['batchsize'],
                           'CG tolerance': config['tol']})

    train_cgd(epoch_num=config['epoch_num'],
              optim_type=config['optimizer'],
              startPoint=chk_path, start_n=start_n,
              show_iter=config['show_iter'], logdir=config['logdir'],
              z_dim=config['z_dim'], batchsize=config['batchsize'],
              l2_penalty=0.0, momentum=config['momentum'],
              data_path=config['datapath'], dataname=config['dataset'],
              loss_name=config['loss_type'],
              model_name=config['model'], model_config=model_args,
              tols=tols, device=device, gpu_num=config['gpu_num'],
              collect_info=True, log=config['log'], ada_train=True, args=config)
