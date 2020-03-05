from optims.cgd_utils import conjugate_gradient, Hvp_vec
import unittest

import torch
import torch.nn as nn
import torch.autograd as autograd


device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'


class NetD(nn.Module):
    def __init__(self):
        super(NetD, self).__init__()
        self.net = nn.Linear(2, 1)
        self.weight_init()

    def forward(self, x):
        return self.net(x)

    def weight_init(self):
        self.net.weight.data = torch.Tensor([[1.0, 2.0]])
        self.net.bias.data = torch.Tensor([-1.0])


class NetG(nn.Module):
    def __init__(self):
        super(NetG, self).__init__()
        self.net = nn.Linear(1, 2)
        self.weight_init()

    def forward(self, x):
        return self.net(x)

    def weight_init(self):
        self.net.weight.data = torch.Tensor([[3.0], [-1.0]])
        self.net.bias.data = torch.Tensor([-4.0, 3.0])


if __name__ == '__main__':
    z = torch.tensor([2.0], device=device)
    D = NetD().to(device)
    G = NetG().to(device)
    print('===discriminator===')
    print(D.net.weight.data)
    print(D.net.bias.data)
    print('===generator===')
    print(G.net.weight.data)
    print(G.net.bias.data)
    x = G(z)
    loss = D(x)

    grad_g = autograd.grad(loss, list(G.parameters()), create_graph=True, retain_graph=True)
    grad_d = autograd.grad(loss, list(D.parameters()), create_graph=True, retain_graph=True)

    grad_g_vec = torch.cat([g.contiguous().view(-1) for g in grad_g])
    grad_d_vec = torch.cat([g.contiguous().view(-1) for g in grad_d])
    print(grad_g_vec)
    print(grad_d_vec)
    grad_g_vec_d = grad_g_vec.clone().detach()
    grad_d_vec_d = grad_d_vec.clone().detach()

    hvp_g_vec = Hvp_vec(grad_d_vec, list(G.parameters()), grad_d_vec_d, retain_graph=True)
    hvp_d_vec = Hvp_vec(grad_g_vec, list(D.parameters()), grad_g_vec_d, retain_graph=True)
    print(hvp_g_vec)
    print(hvp_d_vec)

    cg_g, g_iter_num = conjugate_gradient(grad_x=grad_g_vec, grad_y=grad_d_vec,
                                          x_params=list(G.parameters()), y_params=list(D.parameters()),
                                          b=grad_g_vec_d, lr_x=1.0, lr_y=1.0,
                                          device=device)
    cg_d, d_iter_num = conjugate_gradient(grad_x=grad_d_vec, grad_y=grad_g_vec,
                                          x_params=list(D.parameters()), y_params=list(G.parameters()),
                                          b=grad_d_vec_d, lr_x=1.0, lr_y=1.0,
                                          device=device)
    print(cg_g)
    print(cg_d)
    print('G cg iter num: {}, D cg iter num: {}'.format(g_iter_num, d_iter_num))












