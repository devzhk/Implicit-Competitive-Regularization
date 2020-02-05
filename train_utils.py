import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, LSUN
from GANs.models import dc_D, dc_G, GoodDiscriminator, GoodGenerator, GoodDiscriminatord


class lr_scheduler(object):
    def __init__(self, optimizer, milestone):
        self.optim = optimizer
        self.milestone = milestone

    def step(self, epoch, gamma=10):
        if epoch in self.milestone:
            lr_max, lr_min = self.optim.state['lr_max'], self.optim.state['lr_min']
            lr_max /= gamma
            lr_min /= gamma
            self.optim.set_lr(lr_max=lr_max, lr_min=lr_min)


def get_diff(net, model_vec):
    current_model = torch.cat([p.contiguous().view(-1) for p in net.parameters()]).detach()
    weight_vec = current_model - model_vec
    vom = torch.norm(weight_vec, p=2)
    return vom


def get_model(config):
    model = config['model']
    if model == 'mnist_GAN':
        D = dc_D()
        G = dc_G(z_dim=config['z_dim'])
    elif model == 'cifar-DCGAN':
        D = GoodDiscriminatord() if config['dropout'] else GoodDiscriminator()
        G = GoodGenerator()
    return D, G


def get_data(dataname, path, img_size=64):
    if dataname == 'CIFAR10':
        dataset = CIFAR10(path, train=True,
                          transform=transforms.Compose([transforms.ToTensor(),
                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                        ]),
                          download=True)
    elif dataname == 'MNIST':
        dataset = MNIST(path, train=True,
                        transform=transforms.Compose([transforms.ToTensor(),
                                                      transforms.Normalize((0.5,), (0.5,))
                                                      ]),
                        download=True)
    elif dataname == 'LSUN':
        dataset = LSUN(path, classes=['dining_room_train'], transform=transforms.Compose([
            transforms.Scale(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))
    return dataset


def save_checkpoint(path, name, D, G, optimizer=None):
    chk_name = 'checkpoints/%s/' % path
    if not os.path.exists(chk_name):
        os.makedirs(chk_name)
    d_state_dict = D.state_dict()
    g_state_dict = G.state_dict()
    if optimizer is not None:
        optim_dict = optimizer.state_dict()
    else:
        optim_dict = 0
    torch.save({
        'D': d_state_dict,
        'G': g_state_dict,
        'optim': optim_dict
    }, chk_name + name)
    print('model is saved at %s' % chk_name + name)


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


def model_init(D=None, G=None, init_d=False, init_g=False):
    if D is not None and init_d:
        D.apply(weights_init_d)
        print('initial D with normal')
    if G is not None and init_g:
        G.apply(weights_init_g)
        print('initial G with normal')