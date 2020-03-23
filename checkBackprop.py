import torch
import torch.nn as nn
import torch.autograd as autograd
from optims import ICR, BCGD, testBCGD
from tensorboardX import SummaryWriter
from optims.cgd_utils import Hvp_vec
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
    z = torch.tensor([2.0], device=device)
    real_x = torch.tensor([[3.0, 4.0]], device=device)
    loss = D(G(z)) - D(real_x)

    grad_d = autograd.grad(loss, list(D.parameters()), create_graph=True, retain_graph=True)
    grad_g = autograd.grad(loss, list(G.parameters()), create_graph=True, retain_graph=True)

    g_param = torch.cat([p.contiguous().view(-1) for p in list(G.parameters())])
    d_param = torch.cat([p.contiguous().view(-1) for p in list(D.parameters())])
    print(g_param)
    print(d_param)
    # grad_ggt = torch.tensor([2 * d_param[0].data, 2 * d_param[1].data,
    #                          d_param[0].data, d_param[1].data], device=device)
    # grad_dgt = torch.tensor([2 * g_param[0].data - 3.0 + g_param[2].data,
    #                          2 * g_param[1].data - 4.0 + g_param[3].data, 0.0])

    grad_g_vec = torch.cat([g.contiguous().view(-1) for g in grad_g])
    grad_d_vec = torch.cat([g.contiguous().view(-1) for g in grad_d])
    grad_g_vec_d = grad_g_vec.clone().detach()
    grad_d_vec_d = grad_d_vec.clone().detach()

    hvp_g_vec = Hvp_vec(grad_d_vec, list(G.parameters()), grad_d_vec_d, retain_graph=True)
    hvp_d_vec = Hvp_vec(grad_g_vec, list(D.parameters()), grad_g_vec_d, retain_graph=True)

    hvp_g = torch.tensor([4 * g_param[0].data - 6.0 + 2 * g_param[2].data,
                          4 * g_param[1].data - 8.0 + 2 * g_param[3].data,
                          2 * g_param[0].data - 3.0 + g_param[2].data,
                          2 * g_param[1].data - 4.0 + g_param[3].data])
    hvp_d = torch.tensor([5 * d_param[0].data, 5 * d_param[1].data, 0.0])
    print(hvp_g_vec - hvp_g)
    print(hvp_d_vec - hvp_d)
    # print(grad_g_vec - grad_ggt)
    # print(grad_dgt)
    # print(grad_d_vec)


if __name__ == '__main__':
    train()