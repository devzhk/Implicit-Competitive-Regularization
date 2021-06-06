import os
import time
import yaml

import torch
import torch.nn as nn
import torchvision.utils as vutils

from torch.utils.data import DataLoader

from optims import SACGD
from utils.train_utils import get_data, weights_init_d, weights_init_g, \
    save_checkpoint, get_model
from utils.losses import get_loss
from utils.argparser import cgd_trainer

try:
    import wandb

except ImportError:
    wandb = None

# seed = torch.randint(0, 1000000, (1,))
seed = 2020
torch.manual_seed(seed=seed)
print('random seed : %d' % seed)


def train_cgd(epoch_num=10, optim_type='ACGD',
              startPoint=None, start_n=0,
              z_dim=128,
              tols={'tol': 1e-10, 'atol': 1e-16},
              l2_penalty=0.0,
              loss_name='WGAN', model_name='dc',
              model_config=None,
              data_path='None',
              show_iter=100, logdir='test',
              dataname='CIFAR10',
              device=torch.device('cpu'),
              gpu_num=1, log=False,
              collect_info=False, args=None):
    lr_d = args.lr_d
    lr_g = args.lr_g
    dataset = get_data(dataname=dataname, path=data_path)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batchsize, shuffle=True,
                            num_workers=4)
    D, G = get_model(model_name=model_name, z_dim=z_dim, configs=model_config)
    D.apply(weights_init_d).to(device)
    G.apply(weights_init_g).to(device)
    optimizer = SACGD(x_params=G.parameters(), y_params=D.parameters(),
                      lr_x=lr_g, lr_y=lr_d,
                      tol=tols['tol'], atol=tols['atol'],
                      device=device)
    if startPoint is not None:
        chk = torch.load(startPoint)
        D.load_state_dict(chk['D'])
        G.load_state_dict(chk['G'])
        optimizer.load_state_dict(chk['optim'])
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
            d_loss = get_loss(name=loss_name, g_loss=False,
                              d_real=d_real, d_fake=d_fake,
                              l2_weight=l2_penalty, D=D)
            g_loss = get_loss(name=loss_name, g_loss=True, d_fake=d_fake,
                              l2_weight=l2_penalty, D=D)
            optimizer.zero_grad()
            optimizer.step(lossG=g_loss, lossD=d_loss)

            num_correct = torch.sum(d_real > 0) + torch.sum(d_fake < 0)
            acc = num_correct.item() / (d_real.shape[0] + d_fake.shape[0])
            accs[count % mod] = acc
            acc_indicator = sum(accs) / mod

            if count % show_iter == 0 and count != 0:
                time_cost = time.time() - timer
                print('Iter :%d , G Loss: %.5f, D Loss: %.5f, time: %.3fs'
                      % (count, g_loss.itme(), d_loss.item(), time_cost))
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
                        'D Loss': d_loss.item(),
                        'G_loss': g_loss.item(),
                        'Acc_indicator': acc_indicator
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
    config = parser.parse_args()
    print(config)
    # load model configuration parameters if any
    model_args = None
    if config.model_config is not None:
        with open(config.model_config, 'r') as configFile:
            model_config = yaml.safe_load(configFile)
        model_args = model_config['parameters']

    start_n = config.startn
    chk_path = config.checkpoint
    lr_g = config.lr_g
    lr_d = config.lr_d
    # milestones = {'0': (lr_g, lr_d)}
    tols = {'tol': config.tol, 'atol': config.atol}
    print(tols)

    if wandb and config.log:
        wandb.init(project="%s-ada-sacgd" % config.dataset,
                   config={'lr_g': lr_g,
                           'lr_d': lr_d,
                           'Model name': config.model,
                           'Batchsize': config.batchsize,
                           'CG tolerance': config.tol})

    train_cgd(epoch_num=config.epoch_num,
              optim_type=config.optimizer,
              startPoint=chk_path, start_n=start_n,
              show_iter=config.show_iter, logdir=config.logdir,
              z_dim=config.z_dim,
              l2_penalty=0.0,
              data_path=config.datapath, dataname=config.dataset,
              loss_name=config.loss_type,
              model_name=config.model, model_config=model_args,
              tols=tols, device=device, gpu_num=config.gpu_num,
              collect_info=True, log=config.log, args=config)
