import torch.nn as nn

DIM = 64


class GoodGenerator(nn.Module):
    def __init__(self):
        super(GoodGenerator, self).__init__()
        self.preprocess = nn.Sequential(
            nn.Linear(128, 4 * 4 * 4 * DIM),
            nn.BatchNorm1d(4 * 4 * 4 * DIM),
            nn.ReLU(True),
        )

        self.main_module = nn.Sequential(
            nn.ConvTranspose2d(4 * DIM, 2 * DIM, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2 * DIM),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 * DIM, DIM, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DIM),
            nn.ReLU(True),
            nn.ConvTranspose2d(DIM, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4 * DIM, 4, 4)
        output = self.main_module(output)
        return output.view(-1, 3, 32, 32)


class GoodDiscriminator(nn.Module):
    def __init__(self):
        super(GoodDiscriminator, self).__init__()
        self.main_module = nn.Sequential(
            nn.Conv2d(3, DIM, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            # 16x16
            nn.Conv2d(DIM, 2 * DIM, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            # 8x8
            nn.Conv2d(2 * DIM, 4 * DIM, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            # 4 x 4
        )
        self.linear = nn.Linear(4 * 4 * 4 * DIM, 1)

    def forward(self, input):
        output = self.main_module(input)
        output = output.view(-1, 4 * 4 * 4 * DIM)
        # print(output.shape)
        output = self.linear(output)
        # print(output.shape)
        return output


class GoodDiscriminatord(nn.Module):
    def __init__(self, dropout=0.5):
        super(GoodDiscriminatord, self).__init__()
        self.main_module = nn.Sequential(
            nn.Conv2d(3, DIM, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Dropout2d(dropout),
            # 16x16
            nn.Conv2d(DIM, 2 * DIM, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Dropout2d(dropout),
            # 8x8
            nn.Conv2d(2 * DIM, 4 * DIM, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Dropout2d(dropout),
            # 4 x 4
        )
        self.linear = nn.Linear(4 * 4 * 4 * DIM, 1)

    def forward(self, input):
        output = self.main_module(input)
        output = output.view(-1, 4 * 4 * 4 * DIM)
        # print(output.shape)
        output = self.linear(output)
        # print(output.shape)
        return output


class GoodDiscriminatorbn(nn.Module):
    def __init__(self):
        super(GoodDiscriminatorbn, self).__init__()
        self.main_module = nn.Sequential(
            nn.Conv2d(3, DIM, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DIM),
            nn.LeakyReLU(),
            # nn.Softplus(),
            # nn.Dropout2d(),
            # 16x16
            nn.Conv2d(DIM, 2 * DIM, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2 * DIM),
            nn.LeakyReLU(),
            # nn.Softplus(),
            # nn.Dropout2d(),
            # 8x8
            nn.Conv2d(2 * DIM, 4 * DIM, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4 * DIM),
            nn.LeakyReLU(),
            # nn.Softplus(),
            # nn.Dropout2d(),
            # 4 x 4
        )
        self.linear = nn.Linear(4 * 4 * 4 * DIM, 1)

    def forward(self, input):
        output = self.main_module(input)
        output = output.view(-1, 4 * 4 * 4 * DIM)
        # print(output.shape)
        output = self.linear(output)
        # print(output.shape)
        return output


class dc_d(nn.Module):
    def __init__(self):
        super(dc_d, self).__init__()
        self.conv = nn.Sequential(
            # 3 * 32x32
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2),
            # 32 * 14x14
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2)
            # 64 * 5x5
        )
        self.fc = nn.Sequential(
            nn.Linear(1600, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)


class dc_g(nn.Module):
    def __init__(self, z_dim=96):
        super(dc_g, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 8 * 8 * 128),
            nn.ReLU(),
            nn.BatchNorm1d(8 * 8 * 128),
        )
        self.convt = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2,
                               padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], 128, 8, 8)
        return self.convt(x)


class DC_g(nn.Module):
    def __init__(self, z_dim=100, channel_num=3):
        super(DC_g, self).__init__()
        self.main_module = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            # 1024 * 4x4
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 512 * 8x8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 256 * 16x16
            nn.ConvTranspose2d(256, channel_num, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
            # 3 * 32x32
        )

    def forward(self, input):
        return self.main_module(input)


class DC_d(nn.Module):
    def __init__(self, channel_num=3):
        super(DC_d, self).__init__()
        self.main_module = nn.Sequential(
            nn.Conv2d(channel_num, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            # 1024 * 4x4
            nn.Conv2d(1024, 1, kernel_size=4, stride=2, padding=0),
        )

    def forward(self, input):
        return self.main_module(input)


class DC_generator(nn.Module):
    def __init__(self, z_dim=100, channel_num=3, feature_num=64):
        super(DC_generator, self).__init__()
        self.main_module = nn.Sequential(
            nn.ConvTranspose2d(z_dim, feature_num * 8, kernel_size=4, stride=1, padding=0,
                               bias=False),
            nn.BatchNorm2d(feature_num * 8),
            nn.ReLU(inplace=True),
            # (feature_num * 8) * 4x4
            nn.ConvTranspose2d(feature_num * 8, feature_num * 4, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(feature_num * 4),
            nn.ReLU(inplace=True),
            # (feature_num * 4) * 8x8
            nn.ConvTranspose2d(feature_num * 4, feature_num * 2, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(feature_num * 2),
            nn.ReLU(inplace=True),
            # (feature_num * 2) * 16x16
            nn.ConvTranspose2d(feature_num * 2, feature_num, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(feature_num),
            nn.ReLU(inplace=True),
            # (feature_num * 2) * 32x32
            nn.ConvTranspose2d(feature_num, channel_num, kernel_size=4, stride=2, padding=1,
                               bias=False),
            # channel_num * 64x64
            nn.Tanh()
        )

    def forward(self, input):
        return self.main_module(input)


class DC_discriminator(nn.Module):
    def __init__(self, channel_num=3, feature_num=64):
        super(DC_discriminator, self).__init__()
        self.main_module = nn.Sequential(
            # channel_num * 64x64
            nn.Conv2d(channel_num, feature_num, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (feature_num) * 32x32
            nn.Conv2d(feature_num, feature_num * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_num * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (feature_num * 2) * 16x16
            nn.Conv2d(feature_num * 2, feature_num * 4, kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(feature_num * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (feature_num * 4) * 8x8
            nn.Conv2d(feature_num * 4, feature_num * 8, kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(feature_num * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (feature_num * 8) * 4x4
            nn.Conv2d(feature_num * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # feature_num * 16x16
        )

    def forward(self, input):
        return self.main_module(input)


class DC_discriminatorW(nn.Module):
    def __init__(self, channel_num=3, feature_num=64):
        super(DC_discriminatorW, self).__init__()
        self.main_module = nn.Sequential(
            # channel_num * 64x64
            nn.Conv2d(channel_num, feature_num, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (feature_num) * 32x32
            nn.Conv2d(feature_num, feature_num * 2, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(feature_num * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (feature_num * 2) * 16x16
            nn.Conv2d(feature_num * 2, feature_num * 4, kernel_size=4, stride=2, padding=1,
                      bias=False),
            # nn.BatchNorm2d(feature_num * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (feature_num * 4) * 8x8
            nn.Conv2d(feature_num * 4, feature_num * 8, kernel_size=4, stride=2, padding=1,
                      bias=False),
            # nn.BatchNorm2d(feature_num * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (feature_num * 8) * 4x4
            nn.Conv2d(feature_num * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # feature_num * 16x16
        )

    def forward(self, input):
        return self.main_module(input)


class DC_discriminatord(nn.Module):
    def __init__(self, channel_num=3, feature_num=64):
        super(DC_discriminatord, self).__init__()
        self.main_module = nn.Sequential(
            # channel_num * 64x64
            nn.Conv2d(channel_num, feature_num, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(),
            # (feature_num) * 32x32
            nn.Conv2d(feature_num, feature_num * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_num * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(),
            # (feature_num * 2) * 16x16
            nn.Conv2d(feature_num * 2, feature_num * 4, kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(feature_num * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(),
            # (feature_num * 4) * 8x8
            nn.Conv2d(feature_num * 4, feature_num * 8, kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(feature_num * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(),
            # (feature_num * 8) * 4x4
            nn.Conv2d(feature_num * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # feature_num *
        )

    def forward(self, input):
        return self.main_module(input)


class dc_D(nn.Module):
    def __init__(self):
        super(dc_D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1),  # 32x24x24
            nn.LeakyReLU(0.01),
            # nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),  # 32x12x12
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1), # 64x8x8
            nn.LeakyReLU(0.01),
            # nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2) # 64x4x4
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)


class dc_G(nn.Module):
    def __init__(self, z_dim=96):
        super(dc_G, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 7 * 7 * 128),
            nn.ReLU(),
            nn.BatchNorm1d(7 * 7 * 128),
        )
        self.convt = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2,
                               padding=1),  # 128x7x7 -> 64x14x14
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], 128, 7, 7)
        return self.convt(x)
