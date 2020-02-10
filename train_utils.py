import os
import torch
import torch.nn as nn
import torchvision.utils as vutils

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, LSUN
from GANs.models import dc_D, dc_G, GoodDiscriminator, GoodGenerator, GoodDiscriminatord


mnist_tf = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,))
                               ])


def generate_data(model_weight, path, z_dim=96, device='cpu'):
    chk = torch.load(model_weight)
    print('load from %s' % model_weight)
    dataset = get_data(dataname='MNIST', path='../datas/mnist')
    fixed_z = torch.randn((500, z_dim), device=device)
    fixed_D = dc_D().to(device)
    fixed_G = dc_G(z_dim=z_dim).to(device)
    fixed_D.load_state_dict(chk['D'])
    fixed_G.load_state_dict(chk['G'])
    real_loader = DataLoader(dataset=dataset, batch_size=500, shuffle=True,
                             num_workers=4)
    real_set = next(iter(real_loader))
    real_set = real_set[0].to(device)
    with torch.no_grad():
        fake_set = fixed_G(fixed_z)
        fixed_real_d = fixed_D(real_set)
        fixed_fake_d = fixed_D(fake_set)
        fixed_vec = torch.cat([fixed_real_d, fixed_fake_d])
    if not os.path.exists('figs/select'):
        os.makedirs('figs/select')
    torch.save({'real_set': real_set,
                'fake_set': fake_set,
                'real_d': fixed_real_d,
                'fake_d': fixed_fake_d,
                'pred_vec': fixed_vec}, path)
    for i in range(5):
        j = i * 100
        vutils.save_image(real_set[j: j + 100], 'figs/select/real_set_%d.png' % i, nrow=10, normalize=True)
        vutils.save_image(fake_set[j: j + 100], 'figs/select/fake_set_%d.png' % i, nrow=10, normalize=True)


class lr_scheduler(object):
    def __init__(self, optimizer, milestone):
        self.optim = optimizer
        self.milestone = milestone

    def step(self, epoch, gamma=10):
        if str(epoch) in self.milestone:
            # lr_max, lr_min = self.optim.state['lr_max'], self.optim.state['lr_min']
            self.optim.set_lr(lr_max=self.milestone[0],
                              lr_min=self.milestone[1])


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
    elif dataname == 'LSUN-dining':
        dataset = LSUN(path, classes=['dining_room_train'], transform=transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))
    elif dataname == 'LSUN-bedroom':
        dataset = LSUN(path, classes=['bedroom_train'], transform=transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))
    return dataset


def save_checkpoint(path, name, D, G, optimizer=None):
    chk_name = 'checkpoints/%s/' % path
    if not os.path.exists(chk_name):
        os.makedirs(chk_name)
    try:
        d_state_dict = D.module.state_dict()
        g_state_dict = G.module.state_dict()
    except AttributeError:
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
        nn.init.normal_(m.weight.data, 0.0, 0.01)


def weights_init_g(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.01)


def model_init(D=None, G=None, init_d=False, init_g=False):
    if D is not None and init_d:
        D.apply(weights_init_d)
        print('initial D with normal')
    if G is not None and init_g:
        G.apply(weights_init_g)
        print('initial G with normal')


def select_n_random(data, n=100):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    perm = torch.randperm(len(data))
    return data[perm][:n]