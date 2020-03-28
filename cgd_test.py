from optims.cgd_utils import conjugate_gradient, Hvp_vec
import unittest

import torch
import torch.nn as nn
import torch.autograd as autograd
from optims import BCGD, LCGD
from optims.testOptim import testLCGD, ICR, testBCGD
from tensorboardX import SummaryWriter

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


def train():
    D = NetD().to(device)
    G = NetG().to(device)
    print('===discriminator===')
    print(D.net.weight.data)
    print(D.net.bias.data)
    print('===generator===')
    print(G.net.weight.data)
    print(G.net.bias.data)
    z = torch.tensor([2.0], device=device)
    real_x = torch.tensor([[3.0, 4.0]], device=device)
    loss = D(G(z)) - D(real_x)

    grad_g = autograd.grad(loss, list(G.parameters()), create_graph=True, retain_graph=True)
    grad_d = autograd.grad(loss, list(D.parameters()), create_graph=True, retain_graph=True)
    g_param = torch.cat([p.contiguous().view(-1) for p in list(G.parameters())])
    d_param = torch.cat([p.contiguous().view(-1) for p in list(D.parameters())])
    grad_ggt = torch.tensor([2 * d_param[0].data, 2 * d_param[1].data,
                            d_param[0].data, d_param[1].data], device=device)
    grad_dgt = torch.tensor([2 * g_param[0] - 3.0, 2 * g_param[1] - 4.0, 0.0])

    grad_g_vec = torch.cat([g.contiguous().view(-1) for g in grad_g])
    grad_d_vec = torch.cat([g.contiguous().view(-1) for g in grad_d])
    print(grad_g_vec - grad_ggt)
    print(grad_d_vec - grad_dgt)
    # print(grad_g_vec)
    # print(grad_d_vec)
    grad_g_vec_d = grad_g_vec.clone().detach()
    grad_d_vec_d = grad_d_vec.clone().detach()

    hvp_g_vec = Hvp_vec(grad_d_vec, list(G.parameters()), grad_d_vec_d, retain_graph=True)
    hvp_d_vec = Hvp_vec(grad_g_vec, list(D.parameters()), grad_g_vec_d, retain_graph=True)


if __name__ == '__main__':
    optim_type = 'testBCGD'
    lr = 0.1
    epoch_num = 50
    # device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    D = NetD().to(device)
    G = NetG().to(device)
    writer = SummaryWriter(log_dir='logs/test6/%s-real' % optim_type)
    if optim_type == 'ICR':
        optimizer = ICR(max_params=G.parameters(), min_params=D.parameters(),
                        lr=lr, alpha=1.0, device=device)
    elif optim_type == 'testBCGD':
        optimizer = testBCGD(max_params=G.parameters(), min_params=D.parameters(),
                             lr_max=lr, lr_min=lr, device=device)
    elif optim_type == 'BCGD':
        optimizer = BCGD(max_params=G.parameters(), min_params=D.parameters(),
                         lr_max=lr, lr_min=lr, device=device, solve_x=True)
    elif optim_type == 'LCGD':
        optimizer = LCGD(max_params=G.parameters(), min_params=D.parameters(),
                         lr_max=lr, lr_min=lr, device=device)
    else:
        optimizer = testLCGD(max_params=G.parameters(), min_params=D.parameters(),
                             lr_max=lr, lr_min=lr, device=device)
    for e in range(epoch_num):
        z = torch.tensor([2.0], device=device)
        real_x = torch.tensor([[3.0, 4.0]], device=device)
        loss = D(G(z)) - D(real_x)
        optimizer.zero_grad()
        optimizer.step(loss)
        # if e == 1:
        #     torch.save({'D': D.state_dict(),
        #                 'G': G.state_dict()}, 'net.pth')
        writer.add_scalar('Generator/Weight0', G.net.weight.data[0].item(), global_step=e)
        writer.add_scalar('Generator/Weight1', G.net.weight.data[1].item(), global_step=e)
        writer.add_scalar('Generator/Bias0', G.net.bias.data[0].item(), global_step=e)
        writer.add_scalar('Generator/Bias1', G.net.bias.data[1].item(), global_step=e)
        writer.add_scalar('Discriminator/Weight0', D.net.weight.data[0][0].item(), global_step=e)
        writer.add_scalar('Discriminator/Weight1', D.net.weight.data[0][1].item(), global_step=e)
        writer.add_scalar('Discriminator/Bias0', D.net.bias.data[0].item(), global_step=e)
    writer.close()














