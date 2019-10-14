from torchvision import transforms
import torchvision.datasets as dset
import random
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from tensorboardX import  SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.rmsprop import RMSprop
from optimizers import BCGD, ACGD, MCGD, MACGD
from models import DC_generator, DC_discriminator, DC_d, DC_g, DC_discriminatord

import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'
# modes = ['lcgd', 'cgd', 'SGD', 'Adam', 'RMSProp']
gpu_num = 2
seed = 999
# seed = torch.randint(0, 1000000, (1,))
random.seed(seed)
torch.manual_seed(seed=seed)
print('Random seed : %d' % seed )

tf = transforms.Compose([transforms.Resize(64),
                        transforms.CenterCrop(64),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                         ])

def transform(x):
    x = transforms.ToTensor()(x)
    return (x - 0.5) / 0.5

def detransform(x):
    return (x + 1.0) / 2.0

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.002)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.002)
        nn.init.constant_(m.bias.data, 0)


class CeleA():
    def __init__(self, D, G, device, dataset, noise_shape, z_dim=8, batchsize=256, lr=0.1, show_iter=100, weight_decay=0.0,
                 d_penalty=0.0, g_penalty=0.0, logname='celeba'):
        self.lr = lr
        self.batchsize = batchsize
        self.show_iter = show_iter
        self.device = device
        self.z_dim = z_dim
        self.count = 0
        self.weight_decay = weight_decay
        self.d_penalty = d_penalty
        self.g_penalty = g_penalty
        self.logname = logname
        print('learning rate: %.5f \n'
              'weight decay: %.5f\n'
              'l2 penalty on discriminator: %.5f\n'
              'l2 penalty on generator: %.5f'
              % (self.lr, self.weight_decay, self.d_penalty, self.g_penalty))
        self.dataset = dataset
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=self.batchsize, shuffle=True, num_workers=4)

        self.D = D.to(self.device)
        self.G = G.to(self.device)
        self.D = nn.DataParallel(self.D, list(range(gpu_num)))
        self.G = nn.DataParallel(self.G, list(range(gpu_num)))

        self.D.apply(weights_init)
        self.G.apply(weights_init)

        self.criterion = nn.BCEWithLogitsLoss()
        self.fixed_noise = torch.randn(noise_shape, device=device)

    def l2penalty(self):
        p_d = 0
        p_g = 0
        if self.d_penalty != 0:
            for p in self.D.parameters():
                p_d += torch.dot(p.view(-1), p.view(-1))
        if self.g_penalty != 0:
            for p in self.G.parameters():
                p_g += torch.dot(p.view(-1), p.view(-1))
        return self.d_penalty * p_d - self.g_penalty * p_g

    def load_checkpoint(self, chkpt_path, count, load_d=False, load_g=False):
        self.count = count
        checkpoint = torch.load(chkpt_path)
        if load_d:
            self.D.load_state_dict(checkpoint['D'])
            print('load Discriminator from %s' % chkpt_path)
        if load_g:
            self.G.load_state_dict(checkpoint['G'])
            print('load Generator from %s' % chkpt_path)

    def save_checkpoint(self, path):
        chk_name = 'checkpoints/celeba-%.3f/' % self.lr
        if not os.path.exists(chk_name):
            os.mkdir(chk_name)
        torch.save({
            'D':self.D.state_dict(),
            'G':self.G.state_dict(),
        }, chk_name + path)
        print('save models at %s' % chk_name + path)

    def writer_init(self, logname, comments):
        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        path = ('logs/%s/' % logname) + current_time + '_' + comments
        self.writer = SummaryWriter(logdir=path)

    def show_info(self, timer, D_loss=None, G_loss=None):
        if G_loss is not None:
            print('Iter :%d , D_loss: %.5f, G_loss: %.5f, time: %.3fs' % (self.count, D_loss.item(), G_loss.item(), timer))
        else:
            print('Iter : %d, Loss: %.5f, time: %.3fs' % (self.count, D_loss.item(), timer))
        self.writer.add_scalars('time cost(s)', {'%s iter' % self.show_iter: timer}, self.count)
        z = self.fixed_noise
        fake_data = self.G(z).detach()
        fake_data = detransform(fake_data)
        try:
            self.writer.add_images('Generated images', fake_data, global_step=self.count, dataformats='NCHW')
        except Exception as e:
            print(type(e))
            print('Fail to plot')

    def plot_d(self, d_real, d_fake):
        self.writer.add_scalars('Discriminator output', {'real':d_real.mean().item(), 'fake': d_fake.mean().item()}, self.count)

    def plot_grad(self, gd, gg, hd=None, hg=None, cg_d=None, cg_g=None):
        self.writer.add_scalars('Delta', {'D gradient': gd.item(), 'G gradient': gg.item()}, self.count)
        if hd is not None and hg is not None:
            self.writer.add_scalars('Delta', {'D: lr * hvp': hd.item(), 'G: lr * hvp': hg.item()}, self.count)
        if cg_d is not None and cg_g is not None:
            self.writer.add_scalars('Delta', {'D: cg_d': cg_d.item(), 'G: cg_g': cg_g.item()}, self.count)

    def plot_param(self, D_loss=None, G_loss=None, total_loss=None):
        if G_loss is not None:
            self.writer.add_scalars('Loss', {'D_loss': D_loss.item(), 'G_loss': G_loss.item()}, self.count)
        else:
            self.writer.add_scalars('Loss', {'loss+l2penalty': total_loss.item(), 'loss': D_loss.item()}, self.count)

        wd = torch.norm(torch.cat([p.contiguous().view(-1) for p in self.D.parameters()]), p=2)
        wg = torch.norm(torch.cat([p.contiguous().view(-1) for p in self.G.parameters()]), p=2)

        self.writer.add_scalars('weight', {'D params': wd, 'G params': wg.item()}, self.count)

    def train_gd(self, epoch_num, mode='Adam'):
        print(mode)
        if mode == 'SGD':
            d_optimizer = optim.SGD(self.D.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            g_optimizer = optim.SGD(self.G.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.writer_init(logname=self.logname, comments='SGD-%.3f_%.5f' % (self.lr, self.weight_decay))
        elif mode == 'Adam':
            d_optimizer = optim.Adam(self.D.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=(0.5, 0.999))
            g_optimizer = optim.Adam(self.G.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=(0.5, 0.999))
            self.writer_init(logname=self.logname, comments='ADAM-%.3f_%.5f' % (self.lr, self.weight_decay))
        elif mode == 'RMSProp':
            d_optimizer = RMSprop(self.D.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            g_optimizer = RMSprop(self.G.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.writer_init(logname=self.logname, comments='RMSProp-%.3f_%.5f' % (self.lr, self.weight_decay))
        timer = time.time()

        for e in range(epoch_num):
            for real_x in self.dataloader:
                self.D.zero_grad()
                real_x = real_x[0].to(self.device)
                d_real = self.D(real_x)

                z = torch.randn((self.batchsize, self.z_dim, 1, 1), device=self.device)  ## changed (shape)
                fake_x = self.G(z)
                d_fake = self.D(fake_x)

                # D_loss = gan_loss(d_real, d_fake)
                D_loss = self.criterion(d_real, torch.ones(d_real.shape, device=self.device)) + \
                         self.criterion(d_fake, torch.zeros(d_fake.shape, device=self.device))
                D_loss.backward()
                d_optimizer.step()

                z = torch.randn((self.batchsize, self.z_dim, 1, 1), device=self.device)  ## changed
                fake_x = self.G(z)
                d_fake = self.D(fake_x)
                # G_loss = g_loss(d_fake)
                G_loss = self.criterion(d_fake, torch.ones(d_fake.shape, device=self.device))
                g_optimizer.zero_grad()
                G_loss.backward()
                g_optimizer.step()
                gd = torch.norm(torch.cat([p.grad.contiguous().view(-1) for p in self.D.parameters()]), p=2)
                gg = torch.norm(torch.cat([p.grad.contiguous().view(-1) for p in self.G.parameters()]), p=2)

                self.plot_param(D_loss=D_loss, G_loss=G_loss)
                self.plot_grad(gd=gd, gg=gg)
                self.plot_d(d_real, d_fake)

                if self.count % self.show_iter == 0:
                    self.show_info(timer=time.time() - timer, D_loss=D_loss, G_loss=G_loss)
                    timer = time.time()
                self.count += 1
        self.writer.close()
        self.save_checkpoint('dg-%s-%.3f.pth' % (mode, self.lr))

    def train_cgd(self, epoch_num=100, mode='ACGD', collect_info=False, loss_type='WGAN'):
        timer = time.time()
        self.writer_init(logname=self.logname, comments='%s-%.3fDP%.4fGP%.4f%.5f' % (mode, self.lr, self.d_penalty, self.g_penalty, self.weight_decay))
        if mode == 'BCGD':
            optimizer = BCGD(max_params=list(self.G.parameters()), min_params=list(self.D.parameters()), lr=self.lr,
                         weight_decay=self.weight_decay, device=self.device, solve_x=False, collect_info=collect_info)
        elif mode == 'ACGD':
            optimizer = ACGD(max_params=list(self.G.parameters()), min_params=list(self.D.parameters()), lr=self.lr,
                         weight_decay=self.weight_decay, device=self.device, solve_x=False, collect_info=collect_info)
        elif mode == 'MCGD':
            optimizer = MCGD(max_params=self.G, min_params=self.D, lr=self.lr, device=self.device, solve_x=False,
                             collect_info=collect_info)
        elif mode == 'MACGD':
            optimizer = MACGD(max_params=self.G, min_params=self.D, beta1=0.9, beta2=0.99, lr=self.lr, device=self.device, solve_x=False,
                             collect_info=collect_info)
        for e in range(epoch_num):
            for real_x in self.dataloader:
                real_x = real_x[0].to(self.device)

                d_real = self.D(real_x)

                z = torch.randn((self.batchsize, self.z_dim, 1, 1), device=self.device)
                fake_x = self.G(z)
                d_fake = self.D(fake_x)
                if loss_type == 'JSD':
                    loss = self.criterion(d_real, torch.ones(d_real.shape, device=self.device)) + \
                           self.criterion(d_fake, torch.zeros(d_fake.shape, device=self.device))
                elif loss_type == 'WGAN':
                    loss = d_fake.mean() - d_real.mean()
                lossp = loss + self.l2penalty()
                optimizer.zero_grad()
                optimizer.step(loss=lossp)
                if collect_info:
                    gg, gd, hg, hd, cg_g, cg_d, time_cg, iter_num = optimizer.getinfo()
                    self.plot_param(D_loss=loss, total_loss=lossp)
                    self.plot_grad(gd=gd, gg=gg, hd=hd, hg=hg, cg_d=cg_d, cg_g=cg_g)
                    self.writer.add_scalars('CG iter num', {'converge iter': iter_num}, self.count)
                    self.writer.add_scalars('CG running time', {'A ** -1 * b': time_cg}, self.count)
                self.plot_d(d_real, d_fake)
                if self.count % self.show_iter == 0:
                    self.show_info(D_loss=loss, timer=time.time() - timer)
                    timer = time.time()
                self.count += 1
            if e % 25 == 0:
                self.save_checkpoint('wdg-%s-%.3f_%d.pth' % (mode, self.lr, self.count))
        self.save_checkpoint('wdg-%s-%.3f_%d.pth' % (mode, self.lr, self.count))


def train_celeba():
    modes = ['lcgd', 'cgd', 'SGD', 'Adam', 'RMSProp']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    dataroot = 'datas/celeba'
    images_num = 64
    batch_size = 128
    z_dim = 100
    learning_rate = 0.0001
    Dropout = True
    dataset = dset.ImageFolder(root=dataroot, transform=tf)
    if Dropout:
        D = DC_discriminatord(channel_num=3, feature_num=64)
        print('Dropout: True')
    else:
        D = DC_discriminator(channel_num=3, feature_num=64)
    G = DC_generator(z_dim=z_dim, channel_num=3, feature_num=64)
    trainer = CeleA(D=D, G=G, device=device, dataset=dataset, z_dim=z_dim, batchsize=batch_size, lr=learning_rate,
                         show_iter=100, weight_decay=0.0, d_penalty=0.001, g_penalty=0, noise_shape=(images_num, z_dim, 1, 1), logname='celeba')
    # trainer.train_gd(epoch_num=10, mode=modes[4])
    trainer.train_cgd(epoch_num=25, mode='MCGD', collect_info=True, loss_type='JSD')


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    train_celeba()