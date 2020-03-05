import torch
import torch.nn as nn
from optims import ICR, BCGD


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

cgdType = 'ICR'
device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
z = torch.tensor([2.0], device=device)
D = NetD().to(device)
G = NetG().to(device)

print('===discriminator===')
print(D.net.weight.data)
print(D.net.bias.data)
print('===generator===')
print(G.net.weight.data)
print(G.net.bias.data)

loss = D(G(z))
if cgdType == 'BCGD':
    optimizer = BCGD(max_params=G.parameters(), min_params=D.parameters(),
                     lr_min=1.0, lr_max=1.0, device=device)
    print('BCGD')
else:
    optimizer = ICR(max_params=G.parameters(), min_params=D.parameters(),
                    lr=1.0, alpha=1.0, device=device)
    print('ICR')

optimizer.step(loss)
print('===discriminator===')
print(D.net.weight.data)
print(D.net.bias.data)
print('===generator===')
print(G.net.weight.data)
print(G.net.bias.data)