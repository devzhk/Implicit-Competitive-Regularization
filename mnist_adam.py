import os
import csv
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

from GANs.models import dc_D, dc_G, GoodDiscriminator, GoodGenerator
from optims.adam import Adam, RAdam
from optims.cgd_utils import zero_grad, Hvp_vec, conjugate_gradient
from train_utils import get_data


seed = torch.randint(0, 1000000, (1,))

torch.manual_seed(seed=seed)
print('random seed : %d' % seed)


def transform(x):
    """
    Transform x into a tensor.

    Args:
        x: (array): write your description
    """
    x = transforms.ToTensor()(x)
    return (x - 0.5) / 0.5


def detransform(x):
    """
    Detransformformform ( x )

    Args:
        x: (array): write your description
    """
    return (x + 1.0) / 2.0


def weights_init_d(m):
    """
    Initialize weight weights.

    Args:
        m: (array): write your description
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.005)


def weights_init_g(m):
    """
    Initialize weights.

    Args:
        m: (array): write your description
    """
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
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            D: (int): write your description
            G: (int): write your description
            device: (todo): write your description
            dataset: (todo): write your description
            z_dim: (int): write your description
            batchsize: (int): write your description
            lr_d: (float): write your description
            lr_g: (todo): write your description
            show_iter: (bool): write your description
            weight_decay: (float): write your description
            d_penalty: (str): write your description
            g_penalty: (str): write your description
            noise_shape: (str): write your description
            gp_weight: (int): write your description
            gpu_num: (int): write your description
        """
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
                                     num_workers=4)

        self.D = D.to(self.device)
        self.G = G.to(self.device)
        if gpu_num > 1:
            self.D = nn.DataParallel(self.D, list(range(gpu_num)))
            self.G = nn.DataParallel(self.G, list(range(gpu_num)))
        self.ini_weight()

        self.criterion = nn.BCEWithLogitsLoss()
        self.fixed_noise = torch.randn(noise_shape, device=device)

    def ini_weight(self):
        """
        Apply weighting weights.

        Args:
            self: (todo): write your description
        """
        self.D.apply(weights_init_d)
        self.G.apply(weights_init_g)

    def generate_data(self):
        """
        Generate data from the data.

        Args:
            self: (todo): write your description
        """
        z = torch.randn((self.batchsize, self.z_dim), device=self.device)
        data = self.G(z)
        return data

    def l2penalty(self):
        """
        Compute the diagonal of the kernel.

        Args:
            self: (todo): write your description
        """
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
        """
        Load checkpoint

        Args:
            self: (todo): write your description
            chkpt_path: (str): write your description
            count: (int): write your description
            load_d: (str): write your description
            load_g: (str): write your description
        """
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
        """
        Save checkpoint to disk to disk.

        Args:
            self: (todo): write your description
            path: (str): write your description
            dataset: (todo): write your description
            d_optim: (todo): write your description
            g_optim: (todo): write your description
        """
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
                'D': d_state_dict,
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
        """
        Initialize writer writer.

        Args:
            self: (todo): write your description
            logname: (str): write your description
            comments: (str): write your description
        """
        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        path = ('logs/%s/' % logname) + current_time + '_' + comments
        self.writer = SummaryWriter(log_dir=path)

    def show_info(self, timer, logdir, D_loss=None, G_loss=None):
        """
        Show image info

        Args:
            self: (todo): write your description
            timer: (todo): write your description
            logdir: (str): write your description
            D_loss: (dict): write your description
            G_loss: (dict): write your description
        """
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
        except Exception as e:
            path = 'figs/%s/' % logdir
            if not os.path.exists(path):
                os.makedirs(path)
            vutils.save_image(fake_data, path + 'iter_%d.png' % self.count)
            print('Fail to plot: save images to %s' % path)

    def print_info(self, timer, D_loss=None, G_loss=None):
        """
        Print image info

        Args:
            self: (todo): write your description
            timer: (todo): write your description
            D_loss: (dict): write your description
            G_loss: (dict): write your description
        """
        if G_loss is not None:
            print('Iter :%d , D_loss: %.5f, G_loss: %.5f, time: %.3fs' % (
                self.count, D_loss.item(), G_loss.item(), timer))
        else:
            print('Iter : %d, Loss: %.5f, time: %.3fs' % (self.count, D_loss.item(), timer))
        fake_data = self.G(self.fixed_noise).detach()
        fake_data = detransform(fake_data)
        vutils.save_image(fake_data, 'figs/cifar10/iter-%d.png' % self.count)

    def plot_d(self, d_real, d_fake):
        """
        Plot d_real

        Args:
            self: (todo): write your description
            d_real: (array): write your description
            d_fake: (array): write your description
        """
        self.writer.add_scalars('Discriminator output',
                                {'real': d_real.mean().item(), 'fake': d_fake.mean().item()},
                                self.count)

    def plot_optim(self, d_steps=None, d_updates=None,
                   g_steps=None, g_updates=None,
                   his=False):
        """
        Plot the optimizer.

        Args:
            self: (todo): write your description
            d_steps: (int): write your description
            d_updates: (todo): write your description
            g_steps: (float): write your description
            g_updates: (todo): write your description
            his: (todo): write your description
        """

        if d_steps is not None and d_updates is not None:
            ds_norm = torch.norm(d_steps, p=2)
            du_norm = torch.norm(d_updates, p=2)
            self.writer.add_scalars('Stepsize length', {'Discriminator': ds_norm}, global_step=self.count)
            self.writer.add_scalars('Update length', {'Discriminator': du_norm}, global_step=self.count)
            if his:
                self.writer.add_histogram('Stepsize/Discriminator', d_steps, global_step=self.count)
                self.writer.add_histogram('Update/Discriminator', d_updates, global_step=self.count)

        if g_steps is not None and g_updates is not None:
            gs_norm = torch.norm(g_steps, p=2)
            gu_norm = torch.norm(g_updates, p=2)
            self.writer.add_scalars('Stepsize length', {'Generator': gs_norm}, global_step=self.count)
            self.writer.add_scalars('Update length', {'Generator': gu_norm}, global_step=self.count)
            if his:
                self.writer.add_histogram('Stepsize/Generator', g_steps, global_step=self.count)
                self.writer.add_histogram('Update/Generator', g_updates, global_step=self.count)

    def plot_grad(self, gd, gg, hd=None, hg=None, cg_d=None, cg_g=None):
        """
        Plot gradient plot.

        Args:
            self: (todo): write your description
            gd: (dict): write your description
            gg: (dict): write your description
            hd: (dict): write your description
            hg: (dict): write your description
            cg_d: (dict): write your description
            cg_g: (dict): write your description
        """
        self.writer.add_scalars('Delta', {'D gradient': gd.item(), 'G gradient': gg.item()},
                                self.count)
        if hd is not None and hg is not None:
            self.writer.add_scalars('Delta', {'D: lr * hvp': hd.item(), 'G: lr * hvp': hg.item()},
                                    self.count)
        if cg_d is not None and cg_g is not None:
            self.writer.add_scalars('Delta', {'D: cg_d': cg_d.item(), 'G: cg_g': cg_g.item()},
                                    self.count)

    def plot_param(self, D_loss=None, G_loss=None, total_loss=None):
        """
        Plot the loss function

        Args:
            self: (todo): write your description
            D_loss: (dict): write your description
            G_loss: (dict): write your description
            total_loss: (dict): write your description
        """
        if D_loss is not None:
            self.writer.add_scalars('Loss', {'D_loss': D_loss.item()}, self.count)
        if G_loss is not None:
            self.writer.add_scalars('Loss', {'G_loss': G_loss.item()}, self.count)
        if total_loss is not None:
            self.writer.add_scalars('Loss', {'loss+l2penalty': total_loss.item()}, self.count)

        d_param = torch.cat([p.contiguous().view(-1) for p in self.D.parameters()])
        g_param = torch.cat([p.contiguous().view(-1) for p in self.G.parameters()])
        wd = torch.norm(d_param, p=2)
        wg = torch.norm(g_param, p=2)
        # self.writer.add_histogram('Parameters/Discriminator', d_param, global_step=self.count)
        # self.writer.add_histogram('Parameters/Generator', g_param, global_step=self.count)
        self.writer.add_scalars('weight', {'D params': wd.item(), 'G params': wg.item()}, self.count)

    def plot_proj(self, epoch, loss, model_vec):
        """
        Plots the sum of the given epoch.

        Args:
            self: (todo): write your description
            epoch: (int): write your description
            loss: (todo): write your description
            model_vec: (todo): write your description
        """
        current_model = torch.cat([p.contiguous().view(-1) for p in self.D.parameters()]).detach()
        weight_vec = model_vec - current_model
        model_vec /= torch.norm(weight_vec, p=2)

        d_param = list(self.D.parameters())
        g_param = list(self.G.parameters())
        eta = 1e-2
        import torch.autograd as autograd
        grad_d = autograd.grad(loss, d_param, create_graph=True, retain_graph=True)
        grad_d_vec = torch.cat([g.contiguous().view(-1) for g in grad_d])
        grad_g = autograd.grad(loss, g_param, create_graph=True, retain_graph=True)
        grad_g_vec = torch.cat([g.contiguous().view(-1) for g in grad_g])
        grad_d_vec_detach = grad_d_vec.clone().detach()
        grad_g_vec_detach = grad_g_vec.clone().detach()
        hvp_d_vec = Hvp_vec(grad_g_vec, d_param, grad_g_vec_detach, retain_graph=True)
        p_d = torch.add(grad_d_vec, eta * hvp_d_vec).detach_()
        cg_d, iter_num = conjugate_gradient(grad_x=grad_d_vec, grad_y=grad_g_vec,
                                            x_params=d_param, y_params=g_param, b=p_d,
                                            nsteps=p_d.shape[0], lr_x=eta, lr_y=eta, device=self.device)
        print('CG iter num: %d' % iter_num)
        proj_grad = torch.dot(model_vec, - grad_d_vec_detach)
        proj_hvp = torch.dot(model_vec, - p_d)
        proj_cg = torch.dot(model_vec, - cg_d)
        self.writer.add_scalars('Projection', {'grad': proj_grad,
                                               'hvp': proj_hvp,
                                               'cgd': proj_cg}, global_step=epoch)
        cos_grad = proj_grad / torch.norm(grad_d_vec_detach, p=2)
        cos_hvp = proj_hvp / torch.norm(p_d, p=2)
        cos_cgd = proj_cg / torch.norm(cg_d, p=2)
        self.writer.add_scalars('Cosine', {'grad': cos_grad,
                                           'hvp': cos_hvp,
                                           'cgd': cos_cgd}, global_step=epoch)

    def plot_diff(self, model_vec):
        """
        Plots the difference between the model_vec

        Args:
            self: (todo): write your description
            model_vec: (todo): write your description
        """
        current_model = torch.cat([p.contiguous().view(-1) for p in self.D.parameters()]).detach()
        weight_vec = current_model - model_vec
        vom = torch.norm(weight_vec, p=2)
        self.writer.add_scalar('Distance from checkpoint', vom, self.count)

    def train_gd(self, epoch_num, mode='Adam',
                 dataname='MNIST', logname='MNIST',
                 loss_type='JSD', his_flag=False,
                 info_time=100, compare_weight=None,
                 optim_state=None):
        """
        Train the gpu epoch.

        Args:
            self: (todo): write your description
            epoch_num: (int): write your description
            mode: (todo): write your description
            dataname: (str): write your description
            logname: (str): write your description
            loss_type: (str): write your description
            his_flag: (bool): write your description
            info_time: (todo): write your description
            compare_weight: (bool): write your description
            optim_state: (todo): write your description
        """
        if compare_weight is not None:
            discriminator = dc_D().to(self.device)
            model_weight = torch.load(compare_weight)
            discriminator.load_state_dict(model_weight['D'])
            model_vec = torch.cat([p.contiguous().view(-1) for p in discriminator.parameters()])
            print('Load discriminator from %s' % compare_weight)
        print(mode)
        if mode == 'Adam':
            d_optimizer = Adam(self.D.parameters(), lr=self.lr_d,
                               weight_decay=self.weight_decay, betas=(0.5, 0.999))
            g_optimizer = Adam(self.G.parameters(), lr=self.lr_g,
                               weight_decay=self.weight_decay, betas=(0.5, 0.999))
        elif mode == 'RMSProp':
            d_optimizer = RMSprop(self.D.parameters(), lr=self.lr_d, weight_decay=self.weight_decay)
            g_optimizer = RMSprop(self.G.parameters(), lr=self.lr_g, weight_decay=self.weight_decay)
        self.writer_init(logname=logname,
                         comments='%s-%.3f_%.5f' % (mode, self.lr_d, self.weight_decay))
        if optim_state is not None:
            chk = torch.load(optim_state)
            d_optimizer.load_state_dict(chk['D_optim'])
            g_optimizer.load_state_dict(chk['G_optim'])
            print('load optimizer state')
        timer = time.time()
        flag = False

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
                D_loss = loss + self.l2penalty()
                if compare_weight is not None and self.count % info_time == 0:
                    self.plot_diff(model_vec=model_vec)
                d_optimizer.zero_grad()
                D_loss.backward()
                if self.count % info_time == 0:
                    flag = True
                else:
                    flag = False
                d_steps, d_updates = d_optimizer.step(info=flag)

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
                g_steps, g_updates = g_optimizer.step(info=flag)
                gd = torch.norm(
                    torch.cat([p.grad.contiguous().view(-1) for p in self.D.parameters()]), p=2)
                gg = torch.norm(
                    torch.cat([p.grad.contiguous().view(-1) for p in self.G.parameters()]), p=2)
                if flag:
                    self.plot_optim(d_steps=d_steps, d_updates=d_updates,
                                    g_steps=g_steps, g_updates=g_updates,
                                    his=his_flag)
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

    def train_sgd(self, epoch_num, mode='Adam',
                 dataname='MNIST', logname='MNIST',
                 loss_type='JSD', compare_weight=None, info_time=100):
        """
        Training function.

        Args:
            self: (todo): write your description
            epoch_num: (int): write your description
            mode: (todo): write your description
            dataname: (str): write your description
            logname: (str): write your description
            loss_type: (str): write your description
            compare_weight: (bool): write your description
            info_time: (todo): write your description
        """
        if compare_weight is not None:
            discriminator = dc_D().to(self.device)
            model_weight = torch.load(compare_weight)
            discriminator.load_state_dict(model_weight['D'])
            model_vec = torch.cat([p.contiguous().view(-1) for p in discriminator.parameters()])
            print('Load discriminator from %s' % compare_weight)

        d_optimizer = optim.SGD(self.D.parameters(), lr=self.lr_d, weight_decay=self.weight_decay)
        g_optimizer = optim.SGD(self.G.parameters(), lr=self.lr_g, weight_decay=self.weight_decay)
        self.writer_init(logname=logname,
                         comments='%s-%.3f_%.3f' % (mode, self.lr_d, self.lr_g))
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
                D_loss = loss + self.l2penalty()
                if compare_weight is not None and self.count % info_time == 0:
                    self.plot_diff(model_vec=model_vec)
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

    def train_sgd_d(self, epoch_num, mode='Adam',
                    dataname='MNIST', logname='MNIST',
                    overtrain_path=None, compare_weight=None,
                    info_time=100):
        """
        Train the epoch.

        Args:
            self: (todo): write your description
            epoch_num: (int): write your description
            mode: (todo): write your description
            dataname: (str): write your description
            logname: (str): write your description
            overtrain_path: (str): write your description
            compare_weight: (bool): write your description
            info_time: (todo): write your description
        """
        path = None
        if overtrain_path is not None:
            path = overtrain_path
        if compare_weight is not None:
            path = compare_weight
        if path is not None:
            discriminator = dc_D().to(self.device)
            model_weight = torch.load(path)
            discriminator.load_state_dict(model_weight['D'])
            model_vec = torch.cat([p.contiguous().view(-1) for p in discriminator.parameters()])
            print('Load discriminator from %s' % path)

        print(mode)
        d_optimizer = optim.SGD(self.D.parameters(), lr=self.lr_d, weight_decay=self.weight_decay)
        self.writer_init(logname=logname,
                         comments='SGD-%.3f_%.3f' % (self.lr_d, self.weight_decay))
        timer = time.time()
        d_losses = []
        g_losses = []
        for e in range(epoch_num):
            tol_correct = 0
            tol_loss = 0
            tol_gloss = 0
            for real_x in self.dataloader:
                real_x = real_x[0].to(self.device)
                d_real = self.D(real_x)

                z = torch.randn((real_x.shape[0], self.z_dim),
                                device=self.device)  ## changed (shape)
                fake_x = self.G(z)
                d_fake = self.D(fake_x)

                # D_loss = gan_loss(d_real, d_fake)
                D_loss = self.criterion(d_real, torch.ones(d_real.shape, device=self.device)) + \
                         self.criterion(d_fake, torch.zeros(d_fake.shape, device=self.device))
                tol_loss += D_loss.item() * real_x.shape[0]
                G_loss = self.criterion(d_fake, torch.ones(d_fake.shape, device=self.device)).detach_()
                tol_gloss += G_loss.item() * fake_x.shape[0]
                if self.d_penalty != 0:
                    D_loss += self.l2penalty()
                if overtrain_path is not None and self.count % 2 == 0:
                    self.plot_proj(epoch=self.count, model_vec=model_vec, loss=D_loss)
                if compare_weight is not None and self.count % info_time == 0:
                    self.plot_diff(model_vec=model_vec)
                d_optimizer.zero_grad()
                zero_grad(self.G.parameters())
                D_loss.backward()
                d_optimizer.step()
                tol_correct += (d_real > 0).sum().item() + (d_fake < 0).sum().item()

                gd = torch.norm(
                    torch.cat([p.grad.contiguous().view(-1) for p in self.D.parameters()]), p=2)
                gg = torch.norm(
                    torch.cat([p.grad.contiguous().view(-1) for p in self.G.parameters()]), p=2)
                self.plot_param(D_loss=D_loss, G_loss=G_loss)
                self.plot_grad(gd=gd, gg=gg)
                self.plot_d(d_real, d_fake)
                if self.count % self.show_iter == 0:
                    self.show_info(timer=time.time() - timer,
                                   D_loss=D_loss, G_loss=G_loss,
                                   logdir=logname)
                    timer = time.time()
                self.count += 1
            tol_loss /= len(self.dataset)
            tol_gloss /= len(self.dataset)
            d_losses.append(tol_loss)
            g_losses.append(tol_gloss)
            acc = 50.0 * tol_correct / len(self.dataset)

            self.writer.add_scalar('Train/D_Loss', tol_loss, global_step=e)
            self.writer.add_scalar('Train/G_Loss', tol_gloss, global_step=e)
            self.writer.add_scalar('Train/Accuracy', acc, global_step=e)
            print('Epoch :{}/{}, Acc: {}/{}: {:.3f}%, '
                  'D Loss mean: {:.4f}, G Loss mean: {:.4f}'
                  .format(e, epoch_num, tol_correct, 2 * len(self.dataset), acc,
                          tol_loss, tol_gloss))
            self.save_checkpoint('fixG_%s-%.3f_%d.pth' % (mode, self.lr_d, e),
                                 dataset=dataname)
        self.writer.close()
        return d_losses, g_losses

    def train_d(self, epoch_num, mode='Adam',
                dataname='MNIST', logname='MNIST',
                overtrain_path=None, compare_weight=None,
                his_flag=False, info_time=100,
                optim_state=None):
        """
        Train the model.

        Args:
            self: (todo): write your description
            epoch_num: (int): write your description
            mode: (str): write your description
            dataname: (str): write your description
            logname: (str): write your description
            overtrain_path: (str): write your description
            compare_weight: (bool): write your description
            his_flag: (todo): write your description
            info_time: (todo): write your description
            optim_state: (todo): write your description
        """
        path = None
        if overtrain_path is not None:
            path = overtrain_path
        if compare_weight is not None:
            path = compare_weight
        if path is not None:
            discriminator = dc_D().to(self.device)
            model_weight = torch.load(path)
            discriminator.load_state_dict(model_weight['D'])
            model_vec = torch.cat([p.contiguous().view(-1) for p in discriminator.parameters()])
            print('Load discriminator from %s' % path)

        print(mode)
        if mode == 'SGD':
            d_optimizer = optim.SGD(self.D.parameters(), lr=self.lr_d, weight_decay=self.weight_decay)
            self.writer_init(logname=logname,
                             comments='SGD-%.3f_%.3f' % (self.lr_d, self.weight_decay))
        elif mode == 'Adam':
            d_optimizer = Adam(self.D.parameters(), lr=self.lr_d,
                               weight_decay=self.weight_decay,
                               betas=(0.5, 0.999))
            self.writer_init(logname=logname,
                             comments='ADAM-%.3f_%.5f' % (self.lr_d, self.weight_decay))
        elif mode == 'RMSProp':
            d_optimizer = RMSprop(self.D.parameters(), lr=self.lr_d, weight_decay=self.weight_decay)
            self.writer_init(logname=logname,
                             comments='RMSProp-%.3f_%.5f' % (self.lr_d, self.weight_decay))

        if optim_state is not None:
            chk = torch.load(optim_state)
            d_optimizer.load_state_dict(chk['D_optim'])
            print('load optimizer state')
        timer = time.time()
        d_losses = []
        g_losses = []
        flag = False
        for e in range(epoch_num):
            tol_correct = 0
            tol_loss = 0
            tol_gloss = 0
            for real_x in self.dataloader:
                real_x = real_x[0].to(self.device)
                d_real = self.D(real_x)

                z = torch.randn((real_x.shape[0], self.z_dim),
                                device=self.device)  ## changed (shape)
                fake_x = self.G(z)
                d_fake = self.D(fake_x)

                # D_loss = gan_loss(d_real, d_fake)
                D_loss = self.criterion(d_real, torch.ones(d_real.shape, device=self.device)) + \
                         self.criterion(d_fake, torch.zeros(d_fake.shape, device=self.device))
                tol_loss += D_loss.item() * real_x.shape[0]
                G_loss = self.criterion(d_fake, torch.ones(d_fake.shape, device=self.device)).detach_()
                tol_gloss += G_loss.item() * fake_x.shape[0]
                if self.d_penalty != 0:
                    D_loss += self.l2penalty()
                if overtrain_path is not None and self.count % 2 == 0:
                    self.plot_proj(epoch=self.count, model_vec=model_vec, loss=D_loss)
                if compare_weight is not None and self.count % info_time == 0:
                    self.plot_diff(model_vec=model_vec)
                self.plot_param(D_loss=D_loss, G_loss=G_loss)
                d_optimizer.zero_grad()
                zero_grad(self.G.parameters())
                D_loss.backward()
                # flag = True if self.count % info_time == 0 else False
                if e != 0:
                    d_optimizer.step()
                # d_steps, d_updates = d_optimizer.step(info=flag)
                # if flag:
                #     self.plot_optim(d_steps=d_steps, d_updates=d_updates,
                #                     his=his_flag)

                tol_correct += (d_real > 0).sum().item() + (d_fake < 0).sum().item()

                gd = torch.norm(
                    torch.cat([p.grad.contiguous().view(-1) for p in self.D.parameters()]), p=2)
                gg = torch.norm(
                    torch.cat([p.grad.contiguous().view(-1) for p in self.G.parameters()]), p=2)

                self.plot_grad(gd=gd, gg=gg)
                self.plot_d(d_real, d_fake)
                if self.count % self.show_iter == 0:
                    self.show_info(timer=time.time() - timer,
                                   D_loss=D_loss, G_loss=G_loss,
                                   logdir=logname)
                    timer = time.time()
                self.count += 1
            tol_loss /= len(self.dataset)
            tol_gloss /= len(self.dataset)
            d_losses.append(tol_loss)
            g_losses.append(tol_gloss)
            acc = 50.0 * tol_correct / len(self.dataset)

            self.writer.add_scalar('Train/D_Loss', tol_loss, global_step=e)
            self.writer.add_scalar('Train/G_Loss', tol_gloss, global_step=e)
            self.writer.add_scalar('Train/Accuracy', acc, global_step=e)
            print('Epoch :{}/{}, Acc: {}/{}: {:.3f}%, '
                  'D Loss mean: {:.4f}, G Loss mean: {:.4f}'
                  .format(e, epoch_num, tol_correct, 2 * len(self.dataset), acc,
                          tol_loss, tol_gloss))
            self.save_checkpoint('fixG_%s-%.5f_%d.pth' % (mode, self.lr_d, e),
                                 dataset=dataname)
        self.writer.close()
        return d_losses, g_losses

    def train_simul(self, epoch_num, mode='Adam',
                    dataname='MNIST', logname='MNIST',
                    compare_weight=None, loss_type='JSD',
                    his_flag=False, info_time=100,
                    optim_state=None):
        """
        Train the optimizer.

        Args:
            self: (todo): write your description
            epoch_num: (int): write your description
            mode: (todo): write your description
            dataname: (str): write your description
            logname: (str): write your description
            compare_weight: (bool): write your description
            loss_type: (str): write your description
            his_flag: (bool): write your description
            info_time: (todo): write your description
            optim_state: (todo): write your description
        """
        if compare_weight is not None:
            discriminator = dc_D().to(self.device)
            model_weight = torch.load(compare_weight)
            discriminator.load_state_dict(model_weight['D'])
            model_vec = torch.cat([p.contiguous().view(-1) for p in discriminator.parameters()])
            print('Load discriminator from %s' % compare_weight)
        print(mode)
        if mode == 'SGD':
            d_optimizer = optim.SGD(self.D.parameters(), lr=self.lr_d, weight_decay=self.weight_decay)
            g_optimizer = optim.SGD(self.G.parameters(), lr=self.lr_g, weight_decay=self.weight_decay)
        elif mode == 'Adam':
            d_optimizer = Adam(self.D.parameters(), lr=self.lr_d,
                               weight_decay=self.weight_decay, betas=(0.5, 0.999))
            g_optimizer = Adam(self.G.parameters(), lr=self.lr_g,
                               weight_decay=self.weight_decay, betas=(0.5, 0.999))
        elif mode == 'RMSProp':
            d_optimizer = RMSprop(self.D.parameters(), lr=self.lr_d, weight_decay=self.weight_decay)
            g_optimizer = RMSprop(self.G.parameters(), lr=self.lr_g, weight_decay=self.weight_decay)
        self.writer_init(logname=logname,
                         comments='%s-%.3f_%.5f' % (mode, self.lr_d, self.weight_decay))
        if optim_state is not None:
            chk = torch.load(optim_state)
            d_optimizer.load_state_dict(chk['D_optim'])
            g_optimizer.load_state_dict(chk['G_optim'])
            print('load optimizer state')
        timer = time.time()
        flag = False

        for e in range(epoch_num):
            for real_x in self.dataloader:
                if self.count % info_time == 0:
                    flag = True
                else:
                    flag = False
                real_x = real_x[0].to(self.device)
                d_real = self.D(real_x)

                z = torch.randn((self.batchsize, self.z_dim),
                                device=self.device)  ## changed (shape)
                fake_x = self.G(z)
                fake_x_c = fake_x.clone().detach()
                d_fake = self.D(fake_x)
                if loss_type == 'JSD':
                    G_loss = self.criterion(d_fake, torch.ones(d_fake.shape, device=self.device))
                else:
                    G_loss = - d_fake.mean()
                g_optimizer.zero_grad()
                G_loss.backward()
                g_steps, g_updates = g_optimizer.step(info=flag)

                d_fake_c = self.D(fake_x_c)
                if loss_type == 'JSD':
                    loss = self.criterion(d_real, torch.ones(d_real.shape, device=self.device)) + \
                           self.criterion(d_fake_c, torch.zeros(d_fake_c.shape, device=self.device))
                else:
                    loss = d_fake_c.mean() - d_real.mean()
                D_loss = loss + self.l2penalty()
                if compare_weight is not None and self.count % info_time == 0:
                    self.plot_diff(model_vec=model_vec)
                d_optimizer.zero_grad()
                D_loss.backward()
                d_steps, d_updates = d_optimizer.step(info=flag)

                gd = torch.norm(
                    torch.cat([p.grad.contiguous().view(-1) for p in self.D.parameters()]), p=2)
                gg = torch.norm(
                    torch.cat([p.grad.contiguous().view(-1) for p in self.G.parameters()]), p=2)
                if flag:
                    self.plot_optim(d_steps=d_steps, d_updates=d_updates,
                                    g_steps=g_steps, g_updates=g_updates,
                                    his=his_flag)
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

    def traing(self, epoch_num, mode='Adam', dataname='MNIST', logname='MNIST'):
        """
        Traverse the optimization

        Args:
            self: (todo): write your description
            epoch_num: (int): write your description
            mode: (todo): write your description
            dataname: (str): write your description
            logname: (str): write your description
        """
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
    """
    Train the dataset

    Args:
    """
    modes = ['lcgd', 'cgd', 'SGD', 'Adam', 'RMSProp']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(device)
    print('MNIST')
    lr = 0.01
    z_dim = 96
    D = dc_D()
    G = dc_G(z_dim=z_dim)
    dataset = MNIST('./datas/mnist', download=True, train=True, transform=transform)
    trainer = VisionData(D=D, G=G, device=device, dataset=dataset, z_dim=z_dim, batchsize=128,
                         lr_d=0.01, lr_g=0.01, show_iter=500,
                         weight_decay=0.0, d_penalty=0.0, g_penalty=0, noise_shape=(64, z_dim),
                         gp_weight=0)
    chk_path = 'checkpoints/0.00000MNIST-0.0100/SGD-0.01000_9000.pth'
    trainer.train_sgd(epoch_num=20, mode=modes[2],
                      dataname='MNIST', logname='sgdtest', info_time=100)
    # trainer.train_gd(epoch_num=20, mode=modes[3], dataname='MNIST', logname='test',
    #                  his_flag=False, info_time=100)
    # trainer.load_checkpoint(chk_path, count=0, load_d=True, load_g=True)
    # trainer.train_sgd_d(epoch_num=30, mode=modes[2],
    #                     dataname='MNIST', logname='sgd',
    #                     compare_weight=chk_path, info_time=100)
    # trainer.load_checkpoint('checkpoints/0.00000MNIST-0.0001/Adam-0.00010_9000.pth', count=0, load_d=True, load_g=True)

    # trainer.train_d(epoch_num=20, mode=modes[3], logname='newover', dataname='MNIST',
    #                 compare_weight='checkpoints/0.00000MNIST-0.0001/Adam-0.00010_9000.pth',
    #                 his_flag=True, info_time=250,
    #                 optim_state='checkpoints/0.00000MNIST-0.0001/Adam-0.00010_9000.pth')
    # trainer.train_simul(epoch_num=20, mode=modes[3], logname='newover', dataname='MNIST',
    #                     compare_weight='checkpoints/0.00000MNIST-0.0001/Adam-0.00010_9000.pth',
    #                     his_flag=True, info_time=100)
    # trainer.train_d(epoch_num=3, mode=modes[3], logname='cosine', dataname='MNIST',
    #                 overtrain_path='/checkpoints/0.00000MNIST-0.0050/SGD-0.00500_9000.pth')
    # trainer.load_checkpoint('checkpoints/0.00000MNIST-0.0001/backup/epoch21-D1.pth', count=32000, load_d=True, load_g=True)


def trains(start, end, step, epoch_num,
           model_name, weight_prefix,
           dataname, data_path, preload_path=None):
    """
    Trains the device.

    Args:
        start: (todo): write your description
        end: (int): write your description
        step: (int): write your description
        epoch_num: (int): write your description
        model_name: (str): write your description
        weight_prefix: (str): write your description
        dataname: (str): write your description
        data_path: (str): write your description
        preload_path: (str): write your description
    """
    import pandas as pd
    modes = ['lcgd', 'cgd', 'SGD', 'Adam', 'RMSProp']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    lr = 0.05
    z_dim = 128
    if model_name == 'dc':
        D = dc_D()
        G = dc_G(z_dim=z_dim)
    elif model_name == 'DC':
        D = GoodDiscriminator()
        G = GoodGenerator()

    dataset = get_data(dataname=dataname, path='../datas/%s' % data_path)
    trainer = VisionData(D=D, G=G, device=device, dataset=dataset, z_dim=z_dim, batchsize=128,
                         lr_d=lr, lr_g=lr, show_iter=500,
                         weight_decay=0.0, d_penalty=0.0, g_penalty=0, noise_shape=(64, z_dim),
                         gp_weight=0)
    d_loss_list = []
    g_loss_list = []
    row_names = ['%s%d.pth' % (weight_prefix, i) for i in range(start, end, step)]
    for weight_path in row_names:
        if preload_path is not None:
            num_fcin = trainer.D.linear.in_features
            trainer.D.linear = nn.Linear(num_fcin, 10)
            trainer.load_checkpoint(preload_path, count=0, load_g=False, load_d=True)
            trainer.D.linear = nn.Linear(num_fcin, 1)
            trainer.D.to(device)
            print('Load pretrained discriminator from %s' % preload_path)
        trainer.load_checkpoint(weight_path, count=0, load_g=True, load_d=False)
        d_losses, g_losses = trainer.train_d(epoch_num=epoch_num, mode=modes[2], logname='train_is', dataname='CIFAR')
        d_loss_list.append(d_losses)
        g_loss_list.append(g_losses)
    df = pd.DataFrame(d_loss_list, index=row_names)
    gf = pd.DataFrame(g_loss_list, index=row_names)
    print(df, gf)
    df.to_csv(r'eval_results/preCIFAR_d_loss.csv')
    gf.to_csv(r'eval_results/preCIFAR_g_loss.csv')


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    # train_mnist()
    # trains()
    preload_path = 'checkpoints/pretrain/pretrain.pth'
    chk = 'checkpoints/ACGD/ACGD-0.0100.010_'
    trains(start=30000, end=210000, step=30000, epoch_num=2,
           model_name='DC', weight_prefix=chk, preload_path=preload_path,
           dataname='CIFAR10', data_path='cifar10')
