import os
import time
import yaml

import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from optims import CNAG
from losses import get_loss
from utils import cgd_trainer
from train_utils import get_data, get_model, save_checkpoint, weights_init_d, weights_init_g

try:
    import wandb

except ImportError:
    wandb = None

seed = 2021
torch.manual_seed(seed=seed)
print('random seed : %d' % seed)


def train(epoch_num=10, optim_type='CNAG',
          startPoint=None, start_n=0,
          z_dim=128, batchsize=64,
          l2_penalty=0.0,
          loss_name='WGAN',
          model_name='dc',
          model_config=None,
          data_path='None',
          show_iter=100, logdir='test',
          dataname='CIFAR10',
          device='cpu', gpu_num=1,
          log=False, args=None):
    lr_d = args['lr_d']
    lr_g = args['lr_g']
    dataset = get_data(dataname=dataname, path=data_path)
    dataloader = DataLoader(dataset=dataset, batch_size=batchsize, shuffle=True,
                            num_workers=4)
    D, G = get_model(model_name=model_name, z_dim=z_dim, configs=model_config)
    D.apply(weights_init_d).to(device)
    G.apply(weights_init_g).to(device)

    d_optimizer = CNAG(D.parameters(), lr=lr_d)
    g_optimizer = CNAG(G.parameters(), lr=lr_g)

    if startPoint is not None:
        chk = torch.load(startPoint)
        D.load_state_dict(chk['D'])
        G.load_state_dict(chk['G'])
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
            if 'DCGAN' in model_name:
                z = torch.randn((real_x.shape[0], z_dim, 1, 1), device=device)
            else:
                z = torch.randn((real_x.shape[0], z_dim), device=device)

            d_optimizer.update_param(-1.0)
            g_optimizer.update_param(1.0)
            d_real = D(real_x)
            fake_x = G(z)
            d_fake = D(fake_x)
            d_loss = get_loss(name=loss_name, g_loss=False,
                            d_real=d_real, d_fake=d_fake,
                            l2_weight=l2_penalty, D=D)
            d_optimizer.zero_grad()
            g_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            d_optimizer.update_param(1.0)
            g_optimizer.update_param(-1.0)
            fake_x = G(z)
            d_fake = D(fake_x)
            g_loss = get_loss(name=loss_name, g_loss=True, d_fake=d_fake)
            d_optimizer.zero_grad()
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            if count % show_iter == 0 and count != 0:
                time_cost = time.time() - timer
                print('Iter :%d , D Loss: %.5f, G Loss: %.5f, time: %.3fs'
                      % (count, d_loss.item(), g_loss.item(), time_cost))
                timer = time.time()
                with torch.no_grad():
                    fake_img = G(fixed_noise).detach()
                    path = 'figs/%s_%s/' % (dataname, logdir)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    vutils.save_image(fake_img, path + 'iter_%d.png' % (count + start_n), normalize=True)
                save_checkpoint(path=logdir,
                                name='%s-%s_%d.pth' % (optim_type, model_name, count + start_n),
                                D=D, G=G,
                                optimizer=d_optimizer,
                                g_optimizer=g_optimizer)
            if wandb and log:
                wandb.log(
                    {
                        'Real score': d_real.mean().item(),
                        'Fake score': d_fake.mean().item(),
                        'D Loss': d_loss.item(),
                        'G Loss': g_loss.item(),
                    }
                )
            count += 1


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    parser = cgd_trainer()
    config = vars(parser.parse_args())
    print(config)
    model_args = None
    if config['model_config'] is not None:
        with open(config['model_config'], 'r') as configFile:
            model_config = yaml.safe_load(configFile)
        model_args = model_config['parameters']
    start_n = config['startn']
    chk_path = config['checkpoint']
    lr_g = config['lr_g']
    lr_d = config['lr_d']

    if wandb and config['log']:
        wandb.init(project="%s-CNAG" % config['dataset'],
                   config={'lr_g': lr_g,
                           'lr_d': lr_d,
                           'Model name': config['model'],
                           'Batchsize': config['batchsize']
                           })
    train(epoch_num=config['epoch_num'],
          optim_type=config['optimizer'],
          startPoint=chk_path, start_n=start_n,
          show_iter=config['show_iter'], logdir=config['logdir'],
          z_dim=config['z_dim'], batchsize=config['batchsize'], l2_penalty=0.0,
          data_path=config['datapath'], dataname=config['dataset'],
          loss_name=config['loss_type'],
          model_name=config['model'], model_config=model_args,
          gpu_num=config['gpu_num'], args=config)