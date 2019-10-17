import csv
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models.inception import inception_v3

from models import GoodGenerator, GoodDiscriminator

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def transform(x):
    x = transforms.ToTensor()(x)
    return (x - 0.5) / 0.5


def detransform(x):
    return (x + 1.0) / 2.0


def weights_init_d(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.005)
    # elif classname.find('BatchNorm') != -1:
    #     nn.init.normal_(m.weight.data, 1.0, 0.02)
    #     nn.init.constant_(m.bias.data, 0)


def weights_init_g(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.005)


class WGAN_GP():
    def __init__(self, D, G, device, dataset, z_dim=8, batchsize=256, lr=0.1, show_iter=100,
                 gp_weight=10.0,
                 d_penalty=0.0, d_iter=1, noise_shape=(64, 8)):
        self.lr = lr
        self.batchsize = batchsize
        self.show_iter = show_iter
        self.device = device
        self.z_dim = z_dim
        self.count = 0
        self.gp_weight = gp_weight
        self.d_penalty = d_penalty
        self.d_iter = d_iter
        self.dataloader = DataLoader(dataset=dataset, batch_size=self.batchsize, shuffle=True,
                                     pin_memory=True, drop_last=True)
        print('learning rate: %.5f \n'
              'l2 penalty on discriminator: %.5f\n'
              'gradient penalty weight: %.3f'
              % (self.lr, self.d_penalty, self.gp_weight))
        self.D = D.to(self.device)
        self.G = G.to(self.device)
        self.D.apply(weights_init_d)
        self.G.apply(weights_init_g)
        self.d_optim = optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5, 0.99))
        self.g_optim = optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.99))
        self.criterion = nn.BCEWithLogitsLoss()
        self.fixed_noise = torch.randn(noise_shape, device=device)

    def writer_init(self, logname, comments):
        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        path = ('logs/%s/' % logname) + current_time + '_' + comments
        self.writer = SummaryWriter(logdir=path)

    def save_checkpoint(self, path, dataset):
        chk_name = 'checkpoints/%s-%.5f/' % (dataset, self.lr)
        if not os.path.exists(chk_name):
            os.mkdir(chk_name)
        torch.save({
            'D': self.D.state_dict(),
            'G': self.G.state_dict(),
        }, chk_name + path)
        print('save models at %s' % chk_name + path)

    def generate_data(self):
        z = torch.randn((self.batchsize, self.z_dim), device=self.device)
        data = self.G(z)
        return data

    def l2_penalty(self, d_penalty):
        p_d = 0
        for p in self.D.parameters():
            p_d += torch.dot(p.view(-1), p.view(-1))
        return p_d * d_penalty

    def gradient_penalty(self, real_x, fake_x):
        alpha = torch.randn((self.batchsize, 1, 1, 1), device=self.device)
        alpha = alpha.expand_as(real_x)
        interploted = alpha * real_x.data + (1.0 - alpha) * fake_x.data
        interploted.requires_grad = True
        interploted_d = self.D(interploted)
        gradients = torch.autograd.grad(outputs=interploted_d, inputs=interploted,
                                        grad_outputs=torch.ones(interploted_d.size(),
                                                                device=self.device),
                                        create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(self.batchsize, -1)
        self.writer.add_scalars('Gradients',
                                {'D gradient L2norm': gradients.norm(p=2, dim=1).mean().item()},
                                self.count)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        return self.gp_weight * ((gradients_norm - 1.0) ** 2).mean()

    def get_inception_score(self, batch_num, splits_num=10):
        net = inception_v3(pretrained=True, transform_input=False).eval().to(self.device)
        resize_module = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=True).to(
            self.device)
        preds = np.zeros((self.batchsize * batch_num, 1000))
        for e in range(batch_num):
            imgs = resize_module(self.generate_data())
            pred = F.softmax(net(imgs), dim=1).data.cpu().numpy()
            preds[e * self.batchsize: e * self.batchsize + self.batchsize] = pred
        split_score = []
        chunk_size = preds.shape[0] // splits_num
        for k in range(splits_num):
            pred_chunk = preds[k * chunk_size: k * chunk_size + chunk_size, :]
            kl_score = pred_chunk * (
                        np.log(pred_chunk) - np.log(np.expand_dims(np.mean(pred_chunk, 0), 0)))
            kl_score = np.mean(np.sum(kl_score, 1))
            split_score.append(np.exp(kl_score))
        return np.mean(split_score), np.std(split_score)

    def train_d(self, real_x, gp, d_penalty):
        fake_x = self.generate_data()
        d_real = self.D(real_x)
        d_fake = self.D(fake_x)
        d_loss = d_fake.mean() - d_real.mean()
        if gp:
            gradient_penalty = self.gradient_penalty(real_x=real_x, fake_x=fake_x)
            self.writer.add_scalars('Loss', {'gradient penalty': gradient_penalty.item()},
                                    self.count)
            d_loss = d_loss + gradient_penalty
        if d_penalty is not None:
            l2_penalty = self.l2_penalty(d_penalty)
            self.writer.add_scalars('Loss', {'l2 penalty': l2_penalty.item()}, self.count)
            d_loss = d_loss + l2_penalty

        self.writer.add_scalars('Discriminator output',
                                {'real': d_real.mean(), 'fake': d_fake.mean()}, self.count)
        wd = torch.norm(torch.cat([p.contiguous().view(-1) for p in self.D.parameters()]), p=2)
        wg = torch.norm(torch.cat([p.contiguous().view(-1) for p in self.G.parameters()]), p=2)
        self.writer.add_scalars('weight', {'D params': wd, 'G params': wg.item()}, self.count)

        self.d_optim.zero_grad()
        d_loss.backward()
        self.d_optim.step()
        self.writer.add_scalars('Loss', {'D loss': d_loss.item()}, self.count)
        if self.count % self.show_iter == 0:
            if gp:
                print('Iter: %d, D loss: %.5f, Gradient penalty: %.5f ' % (
                self.count, d_loss.item(), gradient_penalty.item()))
            elif d_penalty is not None:
                print('Iter: %d, D loss: %.5f, l2 penalty: %.5f ' % (
                self.count, d_loss.item(), l2_penalty.item()))
            else:
                print('Iter: %d, D loss: %.5f' % (self.count, d_loss.item()))

    def train_g(self):
        fake_x = self.generate_data()
        d_fake = self.D(fake_x)
        g_loss = - d_fake.mean()
        self.g_optim.zero_grad()
        g_loss.backward()
        self.g_optim.step()
        self.writer.add_scalars('Loss', {'G loss': g_loss.item()}, self.count)
        if self.count % self.show_iter == 0:
            print('Iter: %d, G loss: %.5f ' % (self.count, g_loss.item()))

    def train_epoch(self, epoch_num=10, dirname='WGANGP', dataname='CIFAR10', gp=True,
                    d_penalty=None):
        self.writer_init(logname=dirname, comments=dataname)
        feildnames = ['iter', 'is_mean', 'is_std', 'time']
        path = 'results/wgangp_inception_score.csv'
        f = open(path, 'w')
        self.iswriter = csv.DictWriter(f, feildnames)
        self.iswriter.writeheader()
        start = time.time()
        timer = time.time()
        for e in range(epoch_num):
            for real_x in self.dataloader:
                real_x = real_x[0].to(self.device)
                self.train_d(real_x, gp=gp, d_penalty=d_penalty)
                if self.count % self.d_iter == 0:
                    self.train_g()
                if self.count % self.show_iter == 0:
                    timer = time.time() - timer
                    print('time cost: %.2f' % timer)
                    img = self.G(self.fixed_noise).detach()
                    img = detransform(img)
                    self.writer.add_images('Generated images', img, global_step=self.count)
                    timer = time.time()
                if self.count % 10000 == 0:
                    with torch.no_grad():
                        score_mean, score_std = self.get_inception_score(batch_num=500)
                        np.set_printoptions(precision=4)
                        print('inception score mean: {}, std: {}'.format(score_mean, score_std))
                        self.iswriter.writerow(
                            {'iter': self.count, 'is_mean': score_mean, 'is_std': score_std,
                             'time': time.time() - start})
                        self.writer.add_scalars('Inception scores', {'mean': score_mean},
                                                self.count)
                self.count += 1


def train_cifar():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    print('CIFAR10')
    learning_rate = 0.0001
    batch_size = 64
    z_dim = 128
    D = GoodDiscriminator()
    G = GoodGenerator()
    dataset = CIFAR10(root='datas/cifar10', train=True, transform=transform)
    trainer = WGAN_GP(D=D, G=G, device=device, dataset=dataset, z_dim=z_dim, batchsize=batch_size,
                      lr=learning_rate,
                      show_iter=500, gp_weight=10, d_penalty=0.0, d_iter=5, noise_shape=(64, z_dim))
    trainer.train_epoch(epoch_num=3000, dirname='WGAN-GP', dataname='CIFAR10', gp=True,
                        d_penalty=None)


if __name__ == '__main__':
    train_cifar()
