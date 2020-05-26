import torch.nn as nn
from .layers import SNConv2d, SNLinear

DIM = 64


class GoodSNDiscriminator(nn.Module):
    def __init__(self):
        super(GoodSNDiscriminator, self).__init__()
        self.main_module = nn.Sequential(
            SNConv2d(3, DIM, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(DIM),
            nn.LeakyReLU(),
            # nn.Softplus(),
            # nn.Dropout2d(),
            # 16x16
            SNConv2d(DIM, 2 * DIM, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(2 * DIM),
            nn.LeakyReLU(),
            # nn.Softplus(),
            # nn.Dropout2d(),
            # 8x8
            SNConv2d(2 * DIM, 4 * DIM, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(4 * DIM),
            nn.LeakyReLU(),
            # nn.Softplus(),
            # nn.Dropout2d(),
            # 4 x 4
        )
        self.linear = SNLinear(4 * 4 * 4 * DIM, 1)

    def forward(self, input):
        output = self.main_module(input)
        output = output.view(-1, 4 * 4 * 4 * DIM)
        # print(output.shape)
        output = self.linear(output)
        # print(output.shape)
        return output