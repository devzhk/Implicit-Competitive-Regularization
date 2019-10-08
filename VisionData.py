from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.rmsprop import RMSprop
from optimizers import BCGD, CGD, MCGD, ACGD, OCGD
from models import dc_D, dc_G, dc_d, dc_g, GoodDiscriminator, GoodGenerator, GoodDiscriminatord
from torchvision.models.inception import inception_v3
import csv
import numpy as np
from torch.nn import functional as F

import time
seed = torch.randint(0, 1000000, (1,))
# bad seeds: 850527
# good seeds: 952132, 64843
torch.manual_seed(seed=seed)
print('random seed : %d' % seed )

def transform(x):
    x = transforms.ToTensor()(x)
    return (x - 0.5) / 0.5


def detransform(x):
    return (x + 1.0) / 2.0


def weights_init_d(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.01)


def weights_init_g(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.01)


def zero_grad(net):
    for p in net.parameters():
        if p.grad is not None:
            p.grad.detach()
            p.grad.zero_()


class GAN_trainer():
    def __init__(self, D, G, device, dataset, z_dim=8, batchsize=256, lr=0.1, show_iter=100, weight_decay=0.0,
                 d_penalty=0.0, g_penalty=0.0, noise_shape=(64, 8), gp_weight=10):
        self.lr = lr
        self.batchsize = batchsize
        self.show_iter = show_iter
        self.device = device
        self.z_dim = z_dim
        self.count = 0
        self.weight_decay = weight_decay
        self.d_penalty = d_penalty
        self.g_penalty = g_penalty
        self.gp_weight = gp_weight
        print('learning rate: %.5f \n'
              'weight decay: %.5f\n'
              'l2 penalty on discriminator: %.5f\n'
              'l2 penalty on generator: %.5f\n'
              'gradient penalty weight: %.2f'
              % (self.lr, self.weight_decay, self.d_penalty, self.g_penalty, self.gp_weight))
        self.dataset = dataset
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=self.batchsize, shuffle=True, num_workers=2, drop_last=True)

        self.D = D.to(self.device)
        self.G = G.to(self.device)
        # self.D = nn.DataParallel(self.D, list(range(2)))
        # self.G = nn.DataParallel(self.G, list(range(2)))

        self.D.apply(weights_init_d)
        self.G.apply(weights_init_g)

        self.criterion = nn.BCEWithLogitsLoss()
        self.fixed_noise = torch.randn(noise_shape, device=device)

    def load(self, path):
        checkpoints = torch.load(path)
        self.D.load_state_dict(checkpoints['D'])
        self.G.load_state_dict(checkpoints['G'])

    def gradient_penalty(self, real_x, fake_x):
        alpha = torch.randn((self.batchsize, 1, 1, 1), device=self.device)
        alpha = alpha.expand_as(real_x)
        interploted = alpha * real_x.data + (1.0 - alpha) * fake_x.data
        interploted.requires_grad = True
        interploted_d = self.D(interploted)
        gradients = torch.autograd.grad(outputs=interploted_d, inputs=interploted,
                                        grad_outputs=torch.ones(interploted_d.size(), device=self.device),
                                        create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(self.batchsize, -1)
        self.writer.add_scalars('Gradients',{'D gradient L2norm': gradients.norm(p=2, dim=1).mean().item()}, self.count)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        return self.gp_weight * ((gradients_norm - 1.0) ** 2).mean()

    def generate_data(self):
        z = torch.randn((self.batchsize, self.z_dim), device=self.device)
        data = self.G(z)
        return data

    def get_inception_score(self, batch_num, splits_num=10):
        net = inception_v3(pretrained=True, transform_input=False).eval().to(self.device)
        resize_module = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=True).to(self.device)
        preds = np.zeros((self.batchsize * batch_num, 1000))
        for e in range(batch_num):
            imgs = resize_module(self.generate_data())
            pred = F.softmax(net(imgs), dim=1).data.cpu().numpy()
            preds[e * self.batchsize: e * self.batchsize + self.batchsize] = pred
        split_score = []
        chunk_size = preds.shape[0] // splits_num
        for k in range(splits_num):
            pred_chunk = preds[k * chunk_size: k * chunk_size + chunk_size, :]
            kl_score = pred_chunk * (np.log(pred_chunk) - np.log(np.expand_dims(np.mean(pred_chunk, 0), 0)))
            kl_score = np.mean(np.sum(kl_score, 1))
            split_score.append(np.exp(kl_score))
        return np.mean(split_score), np.std(split_score)

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

    def save_checkpoint(self, path, dataset):
        chk_name = 'checkpoints/%.5f%s-%.4f/' % (self.d_penalty, dataset, self.lr)
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
        feildnames = ['iter', 'is_mean', 'is_std', 'time', 'gradient calls']
        f = open(path + '/inception_score.csv', 'w')
        self.iswriter = csv.DictWriter(f, feildnames)

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
        if D_loss is not None:
            self.writer.add_scalars('Loss', {'D_loss': D_loss.item()}, self.count)
        if G_loss is not None:
            self.writer.add_scalars('Loss', {'G_loss': G_loss.item()}, self.count)
        if total_loss is not None:
            self.writer.add_scalars('Loss', {'loss+l2penalty': total_loss.item()}, self.count)


        wd = torch.norm(torch.cat([p.contiguous().view(-1) for p in self.D.parameters()]), p=2)
        wg = torch.norm(torch.cat([p.contiguous().view(-1) for p in self.G.parameters()]), p=2)

        self.writer.add_scalars('weight', {'D params': wd, 'G params': wg.item()}, self.count)

    def train_gd(self, epoch_num, mode = 'Adam', dataname='MNIST', logname='MNIST', loss_type='JSD'):
        print(mode)
        if mode == 'SGD':
            d_optimizer = optim.SGD(self.D.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            g_optimizer = optim.SGD(self.G.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.writer_init(logname=logname, comments='SGD-%.3f_%.5f' % (self.lr, self.weight_decay))
        elif mode == 'Adam':
            d_optimizer = optim.Adam(self.D.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=(0.5, 0.999))
            g_optimizer = optim.Adam(self.G.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=(0.5, 0.999))
            self.writer_init(logname=logname, comments='ADAM-%.3f_%.5f' % (self.lr, self.weight_decay))
        elif mode == 'RMSProp':
            d_optimizer = RMSprop(self.D.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            g_optimizer = RMSprop(self.G.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.writer_init(logname=logname, comments='RMSProp-%.3f_%.5f' % (self.lr, self.weight_decay))
        self.iswriter.writeheader()
        timer = time.time()
        start = time.time()
        for e in range(epoch_num):
            for real_x in self.dataloader:
                real_x = real_x[0].to(self.device)
                d_real = self.D(real_x)

                z = torch.randn((self.batchsize, self.z_dim), device=self.device) ## changed (shape)
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

                z = torch.randn((self.batchsize, self.z_dim), device=self.device) ## changed
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
                gd = torch.norm(torch.cat([p.grad.contiguous().view(-1) for p in self.D.parameters()]), p=2)
                gg = torch.norm(torch.cat([p.grad.contiguous().view(-1) for p in self.G.parameters()]), p=2)

                self.plot_param(D_loss=D_loss, G_loss=G_loss)
                self.plot_grad(gd=gd, gg=gg)
                self.plot_d(d_real, d_fake)

                if self.count % self.show_iter == 0:
                    self.show_info(timer=time.time() - timer, D_loss=D_loss, G_loss=G_loss)
                    timer = time.time()
                self.count += 1
                if self.count % 2000 == 0:
                    is_mean, is_std = self.get_inception_score(batch_num=500)
                    print(is_mean, is_std)
                    self.iswriter.writerow({'iter': self.count, 'is_mean':is_mean, 'is_std': is_std, 'time': time.time()-start })
                    self.save_checkpoint('%s-%.5f_%d.pth' % (mode, self.lr, self.count), dataset=dataname)
        self.writer.close()
        self.save_checkpoint('DIM64%s-%.5f_%d.pth' % (mode, self.lr, self.count), dataset=dataname)

    def train_bcgd(self, epoch_num=100, mode='BCGD', collect_info=False, dataname='CIFAR', logname='CIFAR', loss_type='JSD'):
        timer = time.time()
        start = time.time()
        if collect_info:
            self.writer_init(logname=logname, comments='%s-%.4fDP%.4fGP%.4f%.5f' % (mode, self.lr, self.d_penalty, self.g_penalty, self.weight_decay))
            self.iswriter.writeheader()
        if mode == 'BCGD':
            optimizer = BCGD(max_params=list(self.G.parameters()), min_params=list(self.D.parameters()), lr=self.lr,
                         weight_decay=self.weight_decay, device=self.device, solve_x=False, collect_info=collect_info)
        elif mode == 'ACGD':
            optimizer = ACGD(max_params=list(self.G.parameters()), min_params=list(self.D.parameters()), lr=self.lr,
                         weight_decay=self.weight_decay, device=self.device, solve_x=False, collect_info=collect_info)
        elif mode == 'MCGD':
            optimizer = MCGD(max_params=self.G, min_params=self.D, lr=self.lr, device=self.device, solve_x=False, collect_info=collect_info)
        for e in range(epoch_num):
            for real_x in self.dataloader:
                real_x = real_x[0].to(self.device)

                d_real = self.D(real_x)

                z = torch.randn((self.batchsize, self.z_dim), device=self.device)
                fake_x = self.G(z)
                d_fake = self.D(fake_x)
                if loss_type == 'JSD':
                    loss = self.criterion(d_real, torch.ones(d_real.shape, device=self.device)) + \
                           self.criterion(d_fake, torch.zeros(d_fake.shape, device=self.device))
                else:
                    loss = d_fake.mean() - d_real.mean()
                if self.gp_weight != 0:
                    loss = loss + self.gradient_penalty(real_x=real_x, fake_x=fake_x)
                lossp = loss + self.l2penalty()
                optimizer.zero_grad()
                optimizer.step(loss=lossp)
                iter_num = -1
                if self.count % self.show_iter == 0:
                    self.show_info(D_loss=loss, timer=time.time() - timer)
                    timer = time.time()
                    self.plot_d(d_real, d_fake)
                if collect_info:
                    gg, gd, hg, hd, cg_g, cg_d, time_cg, iter_num = optimizer.getinfo()
                    self.plot_param(D_loss=loss, total_loss=lossp)
                    self.plot_grad(gd=gd, gg=gg, hd=hd, hg=hg, cg_d=cg_d, cg_g=cg_g)
                    self.writer.add_scalars('CG it'
                                            'er num', {'converge iter': iter_num}, self.count)
                    self.writer.add_scalars('CG running time', {'A ** -1 * b': time_cg}, self.count)

                self.count += 1
                if self.count % 5000 == 0:
                    is_mean, is_std = self.get_inception_score(batch_num=500)
                    print(is_mean, is_std)
                    self.iswriter.writerow({'iter': self.count, 'is_mean':is_mean, 'is_std': is_std,
                                            'time': time.time()-start, 'gradient calls': 2 * iter_num + 4 })
                    self.save_checkpoint('%s-%.5f_%d.pth' % (mode, self.lr, self.count), dataset=dataname)
        self.save_checkpoint('%s-%.5f_%d.pth' % (mode, self.lr, self.count), dataset=dataname)

    def train_d(self, epoch_num, mode = 'Adam', dataname='MNIST', logname='MNIST'):
        print(mode)
        if mode == 'SGD':
            d_optimizer = optim.SGD(self.D.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.writer_init(logname=logname, comments='SGD-%.3f_%.5f' % (self.lr, self.weight_decay))
        elif mode == 'Adam':
            d_optimizer = optim.Adam(self.D.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                                     betas=(0.5, 0.999))
            self.writer_init(logname=logname, comments='ADAM-%.3f_%.5f' % (self.lr, self.weight_decay))
        elif mode == 'RMSProp':
            d_optimizer = RMSprop(self.D.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.writer_init(logname=logname, comments='RMSProp-%.3f_%.5f' % (self.lr, self.weight_decay))
        timer = time.time()
        for e in range(epoch_num):
            for real_x in self.dataloader:
                real_x = real_x[0].to(self.device)
                d_real = self.D(real_x)

                z = torch.randn((self.batchsize, self.z_dim), device=self.device)  ## changed (shape)
                fake_x = self.G(z)
                d_fake = self.D(fake_x)

                # D_loss = gan_loss(d_real, d_fake)
                D_loss = self.criterion(d_real, torch.ones(d_real.shape, device=self.device)) + \
                         self.criterion(d_fake, torch.zeros(d_fake.shape, device=self.device))
                d_optimizer.zero_grad()
                zero_grad(self.G)
                D_loss.backward()
                d_optimizer.step()

                gd = torch.norm(torch.cat([p.grad.contiguous().view(-1) for p in self.D.parameters()]), p=2)
                gg = torch.norm(torch.cat([p.grad.contiguous().view(-1) for p in self.G.parameters()]), p=2)
                self.plot_param(D_loss=D_loss)
                self.plot_grad(gd=gd, gg=gg)
                self.plot_d(d_real, d_fake)
                if self.count % self.show_iter == 0:
                    self.show_info(timer=time.time() - timer, D_loss=D_loss)
                    timer = time.time()
                self.count += 1
                if self.count % 4000 == 0:
                    self.save_checkpoint('fixG_%s-%.5f_%d.pth' % (mode, self.lr, self.count), dataset=dataname)
            self.writer.close()

    def train_g(self, epoch_num, mode = 'Adam', dataname='MNIST', logname='MNIST'):
        print(mode)
        if mode == 'SGD':
            g_optimizer = optim.SGD(self.G.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.writer_init(logname=logname, comments='SGD-%.3f_%.5f' % (self.lr, self.weight_decay))
        elif mode == 'Adam':
            g_optimizer = optim.Adam(self.G.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                                     betas=(0.5, 0.999))
            self.writer_init(logname=logname, comments='ADAM-%.3f_%.5f' % (self.lr, self.weight_decay))
        elif mode == 'RMSProp':
            g_optimizer = RMSprop(self.G.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.writer_init(logname=logname, comments='RMSProp-%.3f_%.5f' % (self.lr, self.weight_decay))
        timer = time.time()
        for e in range(epoch_num):
            z = torch.randn((self.batchsize, self.z_dim), device=self.device)  ## changed
            fake_x = self.G(z)
            d_fake = self.D(fake_x)
            # G_loss = g_loss(d_fake)
            G_loss = self.criterion(d_fake, torch.ones(d_fake.shape, device=self.device))
            g_optimizer.zero_grad()
            zero_grad(self.D)
            G_loss.backward()
            g_optimizer.step()
            gd = torch.norm(torch.cat([p.grad.contiguous().view(-1) for p in self.D.parameters()]), p=2)
            gg = torch.norm(torch.cat([p.grad.contiguous().view(-1) for p in self.G.parameters()]), p=2)
            self.plot_param(G_loss=G_loss)
            self.plot_grad(gd=gd, gg=gg)
            if self.count % self.show_iter == 0:
                self.show_info(timer=time.time() - timer, D_loss=G_loss)
                timer = time.time()
            self.count += 1
            if self.count % 5000 == 0:
                self.save_checkpoint('fixD_%s-%.5f_%d.pth' % (mode, self.lr, self.count), dataset=dataname)
        self.writer.close()

    def traing(self, epoch_num, mode = 'Adam', dataname='MNIST', logname='MNIST'):
        print(mode)
        if mode == 'SGD':
            d_optimizer = optim.SGD(self.D.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            g_optimizer = optim.SGD(self.G.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.writer_init(logname=logname, comments='SGD-%.3f_%.5f' % (self.lr, self.weight_decay))
        elif mode == 'Adam':
            d_optimizer = optim.Adam(self.D.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                                     betas=(0.5, 0.999))
            g_optimizer = optim.Adam(self.G.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                                     betas=(0.5, 0.999))
            self.writer_init(logname=logname, comments='ADAM-%.3f_%.5f' % (self.lr, self.weight_decay))
        elif mode == 'RMSProp':
            d_optimizer = RMSprop(self.D.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            g_optimizer = RMSprop(self.G.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.writer_init(logname=logname, comments='RMSProp-%.3f_%.5f' % (self.lr, self.weight_decay))

        timer = time.time()

        for e in range(epoch_num):
            for real_x in self.dataloader:
                real_x = real_x[0].to(self.device)
                d_real = self.D(real_x)

                z = torch.randn((self.batchsize, self.z_dim), device=self.device)  ## changed (shape)
                fake_x = self.G(z)
                d_fake = self.D(fake_x.detach())

                # D_loss = gan_loss(d_real, d_fake)
                D_loss = self.criterion(d_real, torch.ones(d_real.shape, device=self.device)) + \
                         self.criterion(d_fake, torch.zeros(d_fake.shape, device=self.device))
                # D_loss = d_fake.mean() - d_real.mean()
                d_optimizer.zero_grad()
                D_loss.backward()
                gd = torch.norm(torch.cat([p.grad.contiguous().view(-1) for p in self.D.parameters()]), p=2)

                z = torch.randn((self.batchsize, self.z_dim), device=self.device)  ## changed
                fake_x = self.G(z)
                d_fake = self.D(fake_x)
                # G_loss = g_loss(d_fake)
                G_loss = self.criterion(d_fake, torch.ones(d_fake.shape, device=self.device))
                g_optimizer.zero_grad()
                G_loss.backward()
                g_optimizer.step()
                gg = torch.norm(torch.cat([p.grad.contiguous().view(-1) for p in self.G.parameters()]), p=2)

                self.plot_param(D_loss=D_loss, G_loss=G_loss)
                self.plot_grad(gd=gd, gg=gg)
                self.plot_d(d_real, d_fake)

                if self.count % self.show_iter == 0:
                    self.show_info(timer=time.time() - timer, D_loss=D_loss, G_loss=G_loss)
                    timer = time.time()
                self.count += 1
            if e % 5 == 0:
                self.save_checkpoint('sfixD%s-%.5f_%d.pth' % (mode, self.lr, self.count), dataset=dataname)
        self.writer.close()
        self.save_checkpoint('sfixD%s-%.5f_%d.pth' % (mode, self.lr, self.count), dataset=dataname)

    def train_ocgd(self, epoch_num, update_D, collect_info=True, dataname='MNIST', logname='MNIST'):
        timer = time.time()
        self.writer_init(logname=logname, comments='%.4fDP%.4fGP%.4f%.5f' % (self.lr, self.d_penalty, self.g_penalty, self.weight_decay))
        if update_D:
            optimizer = OCGD(max_params=list(self.G.parameters()), min_params=list(self.D.parameters()), update_min=update_D,
                             device=self.device, collect_info=True)
        else:
            optimizer = OCGD(max_params=list(self.D.parameters()), min_params=list(self.G.parameters()), update_min=True,
                             device=self.device, collect_info=True)
        for e in range(epoch_num):
            for real_x in self.dataloader:
                real_x = real_x[0].to(self.device)
                d_real = self.D(real_x)
                z = torch.randn((self.batchsize, self.z_dim), device=self.device)
                fake_x = self.G(z)
                d_fake = self.D(fake_x)
                if update_D:
                    loss = self.criterion(d_real, torch.ones(d_real.shape, device=self.device)) + \
                           self.criterion(d_fake, torch.zeros(d_fake.shape, device=self.device))
                else:
                    loss = self.criterion(d_fake, torch.ones(d_fake.shape, device=self.device))
                optimizer.zero_grad()
                optimizer.step(loss=loss)
                if collect_info:
                    gg, gd, hg, hd, cg_g, cg_d, time_cg, iter_num = optimizer.getinfo()
                    self.plot_param(D_loss=loss)
                    self.plot_grad(gd=gd, gg=gg)
                    self.writer.add_scalars('CG iter num', {'converge iter': iter_num}, self.count)
                    self.writer.add_scalars('CG running time', {'A ** -1 * b': time_cg}, self.count)
                    if self.count % self.show_iter == 0:
                        self.show_info(D_loss=loss, timer=time.time() - timer)
                        timer = time.time()
                        self.plot_d(d_real, d_fake)
                self.count += 1
                if self.count % 2000 == 0:
                    self.save_checkpoint('%.5f_%d.pth' % (self.lr, self.count), dataset=dataname)
        self.save_checkpoint('%.5f_%d.pth' % (self.lr, self.count), dataset=dataname)
        self.writer.close()


def train_mnist():
    modes = ['lcgd', 'cgd', 'SGD', 'Adam', 'RMSProp']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    print('MNIST')
    learning_rate = 0.0001
    z_dim = 96
    D = dc_D()
    G = dc_G(z_dim=z_dim)
    dataset = MNIST('datas/mnist', train=True, transform=transform)
    trainer = GAN_trainer(D=D, G=G, device=device, dataset=dataset, z_dim=z_dim, batchsize=128, lr=learning_rate, show_iter=200,
                         weight_decay=0.0, d_penalty=0.0, g_penalty=0, noise_shape=(64, z_dim))
    # trainer.train_d(epoch_num=100, mode=modes[3], logname='MNIST2', dataname='MNIST')
    # trainer.load_checkpoint('checkpoints/0.00000MNIST-0.0001/backup/epoch21-D1.pth', count=32000, load_d=True, load_g=True)
    # trainer.load_checkpoint('checkpoints/MNIST-0.0001/backup/fixG_D1_Adam-0.00010_55000.pth', count=55000, load_d=True, load_g=True)
    trainer.load_checkpoint('checkpoints/0.00000MNIST-0.0001/0.00010_50000.pth', count=50000, load_d=True, load_g=True)
    trainer.traing(epoch_num=5, mode=modes[3], logname='MNIST3', dataname='MNIST')
    # trainer.train_ocgd(epoch_num=50, update_D=True, collect_info=True, logname='MNIST3', dataname='MNIST')
    # trainer.train_gd(epoch_num=60, mode=modes[3], logname='MNIST2', dataname='MNIST')
    # trainer.train_cgd(epoch_num=100, mode=modes[1], cg_time=True)
    # trainer.save_checkpoint('wdg-cgd.pth')
    # trainer.train_bcgd(epoch_num=100, mode='BCGD', collect_info=True, dataname='MNIST', logname='MNIST2')


def train_cifar():
    modes = ['lcgd', 'cgd', 'SGD', 'Adam', 'RMSProp']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    print('CIFAR10')
    learning_rate = 0.15
    batch_size = 128
    z_dim = 96
    D = dc_d()
    G = dc_g(z_dim=z_dim)
    dataset = CIFAR10(root='datas/cifar10', train=True, transform=transform)
    trainer = GAN_trainer(D=D, G=G, device=device, dataset=dataset, z_dim=z_dim, batchsize=batch_size, lr=learning_rate,
                         show_iter=500, weight_decay=0.0, d_penalty=0.001, g_penalty=0, noise_shape=(64, z_dim))
    trainer.train_bcgd(epoch_num=100, mode='ACGD', collect_info=True, dataname='CIFAR10', logname='CIFAR10')
    # trainer.train_gd(epoch_num=100, mode=modes[2], dataname='CIFAR10', logname='CIFAR10')


def train_wgan():
    modes = ['lcgd', 'cgd', 'SGD', 'Adam', 'RMSProp']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    print('CIFAR10')
    learning_rate = 0.0001
    batch_size = 64
    z_dim = 128
    dropout = None
    # Dropout: None or 0.5
    if dropout is not None:
        print('dropout!')
        D = GoodDiscriminatord(dropout=dropout)
    else:
        D = GoodDiscriminator()
    G = GoodGenerator()
    dataset = CIFAR10(root='datas/cifar10', download=False, train=True, transform=transform)
    trainer = GAN_trainer(D=D, G=G, device=device, dataset=dataset, z_dim=z_dim, batchsize=batch_size, lr=learning_rate,
                         show_iter=500, weight_decay=0.0, d_penalty=0.0, g_penalty=0, noise_shape=(64, z_dim), gp_weight=0)
    # trainer.load_checkpoint(chkpt_path='')
    trainer.train_bcgd(epoch_num=300, mode='MCGD', collect_info=True, dataname='CIFAR10-WGAN', logname='CIFAR10-WGAN', loss_type='WGAN')
    # Loss type: JSD, WGAN
    # trainer.train_bcgd(epoch_num=120, mode='ACGD', collect_info=True, dataname='CIFAR10-WGAN', logname='CIFAR10-WGAN', loss_type='WGAN')
    # trainer.train_gd(epoch_num=600, mode=modes[3], dataname='CIFAR10-JSD', logname='CIFAR10-JSD', loss_type='JSD')



if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    # train_mnist()
    # train_cifar()
    train_wgan()

