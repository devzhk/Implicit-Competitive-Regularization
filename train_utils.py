import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from GANs.models import dc_D, dc_G, GoodDiscriminator, GoodGenerator, GoodDiscriminatord


def data_model(config):
    if config['dataset'] == 'CIFAR10':
        dataset = CIFAR10(config['datapath'], train=True,
                          transform=transforms.Compose([transforms.ToTensor(),
                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                        ]),
                          download=True)
        D = GoodDiscriminatord() if config['dropout'] else GoodDiscriminator()
        G = GoodGenerator()
    else:
        dataset = MNIST(config['datapath'], train=True,
                        transform=transforms.Compose([transforms.ToTensor(),
                                                      transforms.Normalize(0.5, 0.5)
                                                      ]),
                        download=True)
        D = dc_D()
        G = dc_G(z_dim=config['z_dim'])
    return dataset, D, G


def save_checkpoint(path, name, optimizer, D, G):
    chk_name = 'checkpoints/' + path
    if not os.path.exists(chk_name):
        os.makedirs(chk_name)
    d_state_dict = D.state_dict()
    g_state_dict = G.state_dict()
    optim_dict = optimizer.state_dict()
    torch.save({
        'D': d_state_dict,
        'G': g_state_dict,
        'optim': optim_dict
    }, chk_name + name)
    print('model is saved at %s' % chk_name + name)


def load_checkpoint(path, D, G, load_d, load_g, optim_d, optim_g)


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