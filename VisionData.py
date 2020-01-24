import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

from tensorboardX import SummaryWriter
from torch.optim.rmsprop import RMSprop
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from GANs.models import dc_D, dc_G

from CGDs.cgd_utils import zero_grad

seed = torch.randint(0, 1000000, (1,))
# bad seeds: 850527
# good seeds: 952132, 64843
torch.manual_seed(seed=seed)
print('random seed : %d' % seed)


def transform(x):
    x = transforms.ToTensor()(x)
    return (x - 0.5) / 0.5


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


class VisionData():
    def __init__(self, D, G, device,
                 dataset, z_dim=8, batchsize=256,
                 lr_d=0.1, lr_g=0.1, show_iter=100,
                 weight_decay=0.0,
                 d_penalty=0.0, g_penalty=0.0,
                 noise_shape=(64, 8), gp_weight=10, gpu_num=1):
        self.lr_d = lr_d
        self.lr_g = lr_g
        self.batchsize = batchsize
        self.show_iter = show_iter
        self.device = device
        self.z_dim = z_dim
        self.count = 0
        self.weight_decay = weight_decay
        self.d_penalty = d_penalty
        self.g_penalty = g_penalty
        self.gp_weight = gp_weight
        print('learning rate d: %.5f \n'
              'learning rate g: %.5f \n'
              'weight decay: %.5f\n'
              'l2 penalty on discriminator: %.5f\n'
              'l2 penalty on generator: %.5f\n'
              'gradient penalty weight: %.2f'
              % (self.lr_d, self.lr_g, self.weight_decay, self.d_penalty, self.g_penalty, self.gp_weight))
        self.dataset = dataset
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=self.batchsize, shuffle=True,
                                     num_workers=2, drop_last=True)

        self.D = D.to(self.device)
        self.G = G.to(self.device)
        if gpu_num > 1:
            self.D = nn.DataParallel(self.D, list(range(gpu_num)))
            self.G = nn.DataParallel(self.G, list(range(gpu_num)))

        self.D.apply(weights_init_d)
        self.G.apply(weights_init_g)

        self.criterion = nn.BCEWithLogitsLoss()
        self.fixed_noise = torch.randn(noise_shape, device=device)

    def generate_data(self):
        z = torch.randn((self.batchsize, self.z_dim), device=self.device)
        data = self.G(z)
        return data

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
        # print('load models from %s' % chkpt_path)

    def save_checkpoint(self, path, dataset,
                        d_optim=None, g_optim=None):
        chk_name = './checkpoints/%.5f%s-%.4f/' % (self.d_penalty, dataset, self.lr_d)
        if not os.path.exists(chk_name):
            os.makedirs(chk_name)

        try:
            d_state_dict = self.D.module.state_dict()
            g_state_dict = self.G.module.state_dict()
        except AttributeError:
            d_state_dict = self.D.state_dict()
            g_state_dict = self.G.state_dict()
        if d_optim is not None and g_optim is not None:
            torch.save({
                'D': d_state_dict,
                'G': g_state_dict,
                'D_optim': d_optim,
                'G_optim': g_optim
            }, chk_name + path)
        elif d_optim is not None:
            torch.save({
                'D':d_state_dict,
                'G': g_state_dict,
                'ACGD': d_optim
            }, chk_name + path)
        else:
            torch.save({
                'D': d_state_dict,
                'G': g_state_dict
            }, chk_name + path)
        print('save models at %s' % chk_name + path)

    def writer_init(self, logname, comments):
        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        path = ('logs/%s/' % logname) + current_time + '_' + comments
        self.writer = SummaryWriter(logdir=path)

    def show_info(self, timer, logdir, D_loss=None, G_loss=None):
        if G_loss is not None:
            print('Iter :%d , D_loss: %.5f, G_loss: %.5f, time: %.3fs' % (
            self.count, D_loss.item(), G_loss.item(), timer))
        else:
            print('Iter : %d, Loss: %.5f, time: %.3fs' % (self.count, D_loss.item(), timer))
        self.writer.add_scalars('time cost(s)', {'%s iter' % self.show_iter: timer}, self.count)
        z = self.fixed_noise
        fake_data = self.G(z).detach()
        fake_data = detransform(fake_data)
        try:
            self.writer.add_images('Generated images', fake_data, global_step=self.count,
                                   dataformats='NCHW')
            path = 'figs/%s/' % logdir
            if not os.path.exists(path):
                os.makedirs(path)
            vutils.save_image(fake_data, path + 'iter_%d.png' % self.count)
        except Exception as e:
            print(type(e))
            print('Fail to plot')

    def print_info(self, timer, D_loss=None, G_loss=None):
        if G_loss is not None:
            print('Iter :%d , D_loss: %.5f, G_loss: %.5f, time: %.3fs' % (
            self.count, D_loss.item(), G_loss.item(), timer))
        else:
            print('Iter : %d, Loss: %.5f, time: %.3fs' % (self.count, D_loss.item(), timer))
        fake_data = self.G(self.fixed_noise).detach()
        fake_data = detransform(fake_data)
        vutils.save_image(fake_data, 'figs/cifar10/iter-%d.png' % self.count)

    def plot_d(self, d_real, d_fake):
        self.writer.add_scalars('Discriminator output',
                                {'real': d_real.mean().item(), 'fake': d_fake.mean().item()},
                                self.count)

    def plot_grad(self, gd, gg, hd=None, hg=None, cg_d=None, cg_g=None):
        self.writer.add_scalars('Delta', {'D gradient': gd.item(), 'G gradient': gg.item()},
                                self.count)
        if hd is not None and hg is not None:
            self.writer.add_scalars('Delta', {'D: lr * hvp': hd.item(), 'G: lr * hvp': hg.item()},
                                    self.count)
        if cg_d is not None and cg_g is not None:
            self.writer.add_scalars('Delta', {'D: cg_d': cg_d.item(), 'G: cg_g': cg_g.item()},
                                    self.count)

    def plot_param(self, D_loss=None, G_loss=None, total_loss=None):
        if D_loss is not None:
            self.writer.add_scalars('Loss', {'D_loss': D_loss.item()}, self.count)
        if G_loss is not None:
            self.writer.add_scalars('Loss', {'G_loss': G_loss.item()}, self.count)
        if total_loss is not None:
            self.writer.add_scalars('Loss', {'loss+l2penalty': total_loss.item()}, self.count)

        wd = torch.norm(torch.cat([p.contiguous().view(-1) for p in self.D.parameters()]), p=2)
        wg = torch.norm(torch.cat([p.contiguous().view(-1) for p in self.G.parameters()]), p=2)

        self.writer.add_scalars('weight', {'D params': wd, 'G params': wg.item()}, self.count)

    def train_gd(self, epoch_num, mode='Adam',
                 dataname='MNIST', logname='MNIST',
                 loss_type='JSD'):
        print(mode)
        if mode == 'SGD':
            d_optimizer = optim.SGD(self.D.parameters(), lr=self.lr_d, weight_decay=self.weight_decay)
            g_optimizer = optim.SGD(self.G.parameters(), lr=self.lr_g, weight_decay=self.weight_decay)
        elif mode == 'Adam':
            d_optimizer = optim.Adam(self.D.parameters(), lr=self.lr_d,
                                     weight_decay=self.weight_decay, betas=(0.5, 0.999))
            g_optimizer = optim.Adam(self.G.parameters(), lr=self.lr_g,
                                     weight_decay=self.weight_decay, betas=(0.5, 0.999))
        elif mode == 'RMSProp':
            d_optimizer = RMSprop(self.D.parameters(), lr=self.lr_d, weight_decay=self.weight_decay)
            g_optimizer = RMSprop(self.G.parameters(), lr=self.lr_g, weight_decay=self.weight_decay)
        self.writer_init(logname=logname,
                         comments='%s-%.3f_%.5f' % (mode, self.lr_d, self.weight_decay))
        timer = time.time()
        for e in range(epoch_num):
            for real_x in self.dataloader:
                real_x = real_x[0].to(self.device)
                d_real = self.D(real_x)

                z = torch.randn((self.batchsize, self.z_dim),
                                device=self.device)  ## changed (shape)
                fake_x = self.G(z)
                d_fake = self.D(fake_x.detach())
                if loss_type == 'JSD':
                    loss = self.criterion(d_real, torch.ones(d_real.shape, device=self.device)) + \
                           self.criterion(d_fake, torch.zeros(d_fake.shape, device=self.device))
                else:
                    loss = d_fake.mean() - d_real.mean()

                # D_loss = gan_loss(d_real, d_fake)
                # D_loss = self.criterion(d_real, torch.ones(d_real.shape, device=self.device)) + \
                #          self.criterion(d_fake, torch.zeros(d_fake.shape, device=self.device))
                D_loss = loss + self.l2penalty()
                d_optimizer.zero_grad()
                D_loss.backward()
                d_optimizer.step()

                z = torch.randn((self.batchsize, self.z_dim), device=self.device)  ## changed
                fake_x = self.G(z)
                d_fake = self.D(fake_x)
                # G_loss = g_loss(d_fake)
                if loss_type == 'JSD':
                    G_loss = self.criterion(d_fake, torch.ones(d_fake.shape, device=self.device))
                else:
                    G_loss = - d_fake.mean()
                g_optimizer.zero_grad()
                G_loss.backward()
                g_optimizer.step()
                gd = torch.norm(
                    torch.cat([p.grad.contiguous().view(-1) for p in self.D.parameters()]), p=2)
                gg = torch.norm(
                    torch.cat([p.grad.contiguous().view(-1) for p in self.G.parameters()]), p=2)

                self.plot_param(D_loss=D_loss, G_loss=G_loss)
                self.plot_grad(gd=gd, gg=gg)
                self.plot_d(d_real, d_fake)

                if self.count % self.show_iter == 0:
                    self.show_info(timer=time.time() - timer, D_loss=D_loss, G_loss=G_loss, logdir=logname)
                    timer = time.time()
                    self.save_checkpoint('%s-%.5f_%d.pth' % (mode, self.lr_d, self.count), dataset=dataname,
                                         d_optim=d_optimizer.state_dict(), g_optim=g_optimizer.state_dict())
                self.count += 1
        self.writer.close()

    def train_d(self, epoch_num, mode='Adam', dataname='MNIST', logname='MNIST'):
        print(mode)
        if mode == 'SGD':
            d_optimizer = optim.SGD(self.D.parameters(), lr=self.lr_d, weight_decay=self.weight_decay)
            self.writer_init(logname=logname,
                             comments='SGD-%.3f_%.5f' % (self.lr_d, self.weight_decay))
        elif mode == 'Adam':
            d_optimizer = optim.Adam(self.D.parameters(), lr=self.lr_d,
                                     weight_decay=self.weight_decay,
                                     betas=(0.5, 0.999))
            self.writer_init(logname=logname,
                             comments='ADAM-%.3f_%.5f' % (self.lr_d, self.weight_decay))
        elif mode == 'RMSProp':
            d_optimizer = RMSprop(self.D.parameters(), lr=self.lr_d, weight_decay=self.weight_decay)
            self.writer_init(logname=logname,
                             comments='RMSProp-%.3f_%.5f' % (self.lr_d, self.weight_decay))
        timer = time.time()
        for e in range(epoch_num):
            for real_x in self.dataloader:
                real_x = real_x[0].to(self.device)
                d_real = self.D(real_x)

                z = torch.randn((self.batchsize, self.z_dim),
                                device=self.device)  ## changed (shape)
                fake_x = self.G(z)
                d_fake = self.D(fake_x)

                # D_loss = gan_loss(d_real, d_fake)
                D_loss = self.criterion(d_real, torch.ones(d_real.shape, device=self.device)) + \
                         self.criterion(d_fake, torch.zeros(d_fake.shape, device=self.device))
                d_optimizer.zero_grad()
                zero_grad(self.G.parameters())
                D_loss.backward()
                d_optimizer.step()

                gd = torch.norm(
                    torch.cat([p.grad.contiguous().view(-1) for p in self.D.parameters()]), p=2)
                gg = torch.norm(
                    torch.cat([p.grad.contiguous().view(-1) for p in self.G.parameters()]), p=2)
                self.plot_param(D_loss=D_loss)
                self.plot_grad(gd=gd, gg=gg)
                self.plot_d(d_real, d_fake)
                if self.count % self.show_iter == 0:
                    self.show_info(timer=time.time() - timer, D_loss=D_loss, logdir=logname)
                    timer = time.time()
                    self.save_checkpoint('fixG_%s-%.5f_%d.pth' % (mode, self.lr_d, self.count),
                                         dataset=dataname)
                self.count += 1
            self.writer.close()

    def traing(self, epoch_num, mode='Adam', dataname='MNIST', logname='MNIST'):
        print(mode)
        if mode == 'SGD':
            d_optimizer = optim.SGD(self.D.parameters(), lr=self.lr_d, weight_decay=self.weight_decay)
            g_optimizer = optim.SGD(self.G.parameters(), lr=self.lr_d, weight_decay=self.weight_decay)
            self.writer_init(logname=logname,
                             comments='SGD-%.3f_%.5f' % (self.lr_d, self.weight_decay))
        elif mode == 'Adam':
            d_optimizer = optim.Adam(self.D.parameters(), lr=self.lr_d,
                                     weight_decay=self.weight_decay,
                                     betas=(0.5, 0.999))
            g_optimizer = optim.Adam(self.G.parameters(), lr=self.lr_d,
                                     weight_decay=self.weight_decay,
                                     betas=(0.5, 0.999))
            self.writer_init(logname=logname,
                             comments='ADAM-%.3f_%.5f' % (self.lr_d, self.weight_decay))
        elif mode == 'RMSProp':
            d_optimizer = RMSprop(self.D.parameters(), lr=self.lr_d, weight_decay=self.weight_decay)
            g_optimizer = RMSprop(self.G.parameters(), lr=self.lr_d, weight_decay=self.weight_decay)
            self.writer_init(logname=logname,
                             comments='RMSProp-%.3f_%.5f' % (self.lr_d, self.weight_decay))

        timer = time.time()

        for e in range(epoch_num):
            for real_x in self.dataloader:
                real_x = real_x[0].to(self.device)
                d_real = self.D(real_x)

                z = torch.randn((self.batchsize, self.z_dim),
                                device=self.device)  ## changed (shape)
                fake_x = self.G(z)
                d_fake = self.D(fake_x.detach())

                # D_loss = gan_loss(d_real, d_fake)
                D_loss = self.criterion(d_real, torch.ones(d_real.shape, device=self.device)) + \
                         self.criterion(d_fake, torch.zeros(d_fake.shape, device=self.device))
                # D_loss = d_fake.mean() - d_real.mean()
                d_optimizer.zero_grad()
                D_loss.backward()
                gd = torch.norm(
                    torch.cat([p.grad.contiguous().view(-1) for p in self.D.parameters()]), p=2)

                z = torch.randn((self.batchsize, self.z_dim), device=self.device)  ## changed
                fake_x = self.G(z)
                d_fake = self.D(fake_x)
                # G_loss = g_loss(d_fake)
                G_loss = self.criterion(d_fake, torch.ones(d_fake.shape, device=self.device))
                g_optimizer.zero_grad()
                G_loss.backward()
                g_optimizer.step()
                gg = torch.norm(
                    torch.cat([p.grad.contiguous().view(-1) for p in self.G.parameters()]), p=2)

                self.plot_param(D_loss=D_loss, G_loss=G_loss)
                self.plot_grad(gd=gd, gg=gg)
                self.plot_d(d_real, d_fake)

                if self.count % self.show_iter == 0:
                    self.show_info(timer=time.time() - timer, D_loss=D_loss, G_loss=G_loss)
                    timer = time.time()
                    self.save_checkpoint('sfixD%s-%.5f_%d.pth' % (mode, self.lr_d, self.count),
                                         dataset=dataname)
                self.count += 1
        self.writer.close()


def train_mnist():
    modes = ['lcgd', 'cgd', 'SGD', 'Adam', 'RMSProp']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    print('MNIST')
    lr = 0.001
    z_dim = 96
    D = dc_D()
    G = dc_G(z_dim=z_dim)
    dataset = MNIST('./datas/mnist', download=True, train=True, transform=transform)
    trainer = VisionData(D=D, G=G, device=device, dataset=dataset, z_dim=z_dim, batchsize=128,
                         lr_d=lr, lr_g=lr, show_iter=600,
                         weight_decay=0.0, d_penalty=0.0, g_penalty=0, noise_shape=(64, z_dim),
                         gp_weight=0)
    # trainer.train_gd(epoch_num=20, mode=modes[3], dataname='MNIST', logname='chk')
    trainer.load_checkpoint('checkpoints/0.00000MNIST-0.0001/Adam-0.00010_600.pth', count=0, load_d=True, load_g=True)

    trainer.train_d(epoch_num=100, mode=modes[3], logname='overtrain', dataname='MNIST')
    # trainer.load_checkpoint('checkpoints/0.00000MNIST-0.0001/backup/epoch21-D1.pth', count=32000, load_d=True, load_g=True)
    # trainer.load_checkpoint('checkpoints/MNIST-0.0001/backup/fixG_D1_Adam-0.00010_55000.pth', count=55000, load_d=True, load_g=True)
    # trainer.load_checkpoint('checkpoints/0.00000MNIST-0.0001/0.00010_50000.pth', count=50000,
    #                         load_d=True, load_g=True)
    # trainer.traing(epoch_num=5, mode=modes[3], logname='MNIST3', dataname='MNIST')
    # trainer.train_ocgd(epoch_num=50, update_D=True, collect_info=True, logname='MNIST3', dataname='MNIST')
    # trainer.train_gd(epoch_num=60, mode=modes[3], logname='MNIST2', dataname='MNIST')
    # trainer.train_cgd(epoch_num=100, mode=modes[1], cg_time=True)
    # trainer.save_checkpoint('wdg-cgd.pth')
    # trainer.train_bcgd(epoch_num=100, mode='BCGD', collect_info=True, dataname='MNIST', logname='MNIST2')


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    train_mnist()


