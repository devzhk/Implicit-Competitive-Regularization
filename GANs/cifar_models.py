import torch.nn as nn


class dcD32(nn.Module):
    def __init__(self):
        """
        Initialize the convolutional layer.

        Args:
            self: (todo): write your description
        """
        super(dcD32, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1),  # 3x32x32 -> 32x28x28
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2),  # 32x14x14
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1),  # 32x14x14 -> 64x10x10
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2),  # 64x5x5
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 5 * 5, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        """
        Forward computation

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)


class dcG32(nn.Module):
    def __init__(self, z_dim=128):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            z_dim: (int): write your description
        """
        super(dcG32, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 8 * 8 * 128)
        )
        self.convt = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64,
                               kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(in_channels=64, out_channels=3,
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        x = self.fc(x)
        x = x.view(x.shape[0], 128, 8, 8)
        return self.convt(x)