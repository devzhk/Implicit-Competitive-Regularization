import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, num_filters, resample=None, batchnorm=True, inplace=False):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            num_filters: (int): write your description
            resample: (int): write your description
            batchnorm: (todo): write your description
            inplace: (todo): write your description
        """
        super(ResBlock, self).__init__()

        if resample == 'up':
            conv_list = [nn.ConvTranspose2d(num_filters, num_filters, 4, stride=2, padding=1),
                        nn.Conv2d(num_filters, num_filters, 3, padding=1)]
            self.conv_shortcut =  nn.ConvTranspose2d(num_filters, num_filters, 1, stride=2, output_padding=1)

        elif resample == 'down':
            conv_list = [nn.Conv2d(num_filters, num_filters, 3, padding=1),
                        nn.Conv2d(num_filters, num_filters, 3, stride=2, padding=1)]
            self.conv_shortcut = nn.Conv2d(num_filters, num_filters, 1, stride=2)

        elif resample == None:
            conv_list = [nn.Conv2d(num_filters, num_filters, 3, padding=1),
                        nn.Conv2d(num_filters, num_filters, 3, padding=1)]
            self.conv_shortcut = None
        else:
            raise ValueError('Invalid resample value.')

        self.block = []
        for conv in conv_list:
            if batchnorm:
                self.block.append(nn.BatchNorm2d(num_filters))
            self.block.append(nn.ReLU(inplace))
            self.block.append(conv)

        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        """
        Forward function fortran.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        shortcut = x
        if not self.conv_shortcut is None:
            shortcut = self.conv_shortcut(x)
        return shortcut + self.block(x)


class ResNet32Generator(nn.Module):
    def __init__(self, z_dim, num_filters=128, batchnorm=True):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            z_dim: (int): write your description
            num_filters: (int): write your description
            batchnorm: (todo): write your description
        """
        super(ResNet32Generator, self).__init__()
        self.num_filters = num_filters

        self.input = nn.Linear(z_dim, 4*4*num_filters)
        self.network = [ResBlock(num_filters, resample='up', batchnorm=batchnorm, inplace=True),
                        ResBlock(num_filters, resample='up', batchnorm=batchnorm, inplace=True),
                        ResBlock(num_filters, resample='up', batchnorm=batchnorm, inplace=True)]
        if batchnorm:
            self.network.append(nn.BatchNorm2d(num_filters))
        self.network += [nn.ReLU(True),
                        nn.Conv2d(num_filters, 3, 3, padding=1),
                        nn.Tanh()]

        self.network = nn.Sequential(*self.network)

    def forward(self, z):
        """
        R forward forward computation.

        Args:
            self: (todo): write your description
            z: (todo): write your description
        """
        x = self.input(z).view(len(z), self.num_filters, 4, 4)
        return self.network(x)


class ResNet32Discriminator(nn.Module):
    def __init__(self, n_in, num_filters=128, batchnorm=False):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            n_in: (int): write your description
            num_filters: (int): write your description
            batchnorm: (todo): write your description
        """
        super(ResNet32Discriminator, self).__init__()

        self.block1 = nn.Sequential(nn.Conv2d(n_in, num_filters, 3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(num_filters, num_filters, 3, stride=2, padding=1))

        self.shortcut1 = nn.Conv2d(n_in, num_filters, 1, stride=2)

        self.network = nn.Sequential(ResBlock(num_filters, resample='down', batchnorm=batchnorm),
                                    ResBlock(num_filters, resample=None, batchnorm=batchnorm),
                                    ResBlock(num_filters, resample=None, batchnorm=batchnorm),
                                    nn.ReLU())
        self.output = nn.Linear(num_filters, 1)

    def forward(self, x):
        """
        Calculate the network.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        y = self.block1(x)
        y = self.shortcut1(x) + y
        y = self.network(y).mean(-1).mean(-1)
        y = self.output(y)
        return y