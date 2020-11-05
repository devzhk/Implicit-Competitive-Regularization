import torch
import torch.nn as nn


class DCGAN_D(nn.Module):
    def __init__(self, insize=64, channel_num=3, feature_num=128, n_extra_layers=0):
        """
        Initialize the module.

        Args:
            self: (todo): write your description
            insize: (int): write your description
            channel_num: (int): write your description
            feature_num: (int): write your description
            n_extra_layers: (list): write your description
        """
        super(DCGAN_D, self).__init__()
        assert insize % 16 == 0, "input size has to be a multiple of 16"

        main = nn.Sequential()
        # input: channel_num x insize x insize
        main.add_module('initial:{0}-{1}:conv'.format(channel_num, feature_num),
                        nn.Conv2d(channel_num, feature_num,
                                  4, 2, 1, bias=False))
        main.add_module('initial:{0}:relu'.format(feature_num),
                        nn.LeakyReLU(0.2, inplace=True))
        # csize: current feature size of conv layer
        # cndf: current feature channels of conv layer
        csize, cndf = insize / 2, feature_num

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}:{1}:conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}:{1}:batchnorm'.format(t, cndf),
                            nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}:{1}:relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid:{0}-{1}:conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid:{0}:batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid:{0}:relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. cndf x 4 x 4 -> 1x1
        main.add_module('final:{0}-{1}:conv'.format(cndf, 1),
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.main = main

    def forward(self, x):
        """
        Eval on x.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        output = self.main(x)
        return output


class DCGAN_G(nn.Module):
    def __init__(self, outsize=64, z_dim=100, nc=3, feature_num=128, n_extra_layers=0):
        '''
        nc: output channel number
        '''
        super(DCGAN_G, self).__init__()
        assert outsize % 16 == 0, "insize has to be a multiple of 16"

        cngf, target_size = feature_num // 2, 4
        while target_size != outsize:
            cngf = cngf * 2
            target_size = target_size * 2

        main = nn.Sequential()
        # latent variable is  z_dim x 1 x 1
        # after initial convolution: cngf x 4 x 4
        main.add_module('initial:{0}-{1}:convt'.format(z_dim, cngf),
                        nn.ConvTranspose2d(z_dim, cngf, 4, 1, 0, bias=False))
        main.add_module('initial:{0}:batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial:{0}:relu'.format(cngf),
                        nn.ReLU(True))
        # csize: current feature size
        # cngf: current number of feature channel
        csize= 4
        while csize < outsize // 2:
            main.add_module('pyramid:{0}-{1}:convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid:{0}:batchnorm'.format(cngf // 2),
                            nn.BatchNorm2d(cngf // 2))
            main.add_module('pyramid:{0}:relu'.format(cngf // 2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}:{1}:conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}:{1}:batchnorm'.format(t, cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}:{1}:relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final:{0}-{1}:convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final:{0}:tanh'.format(nc),
                        nn.Tanh())
        self.main = main

    def forward(self, x):
        """
        Eval on x.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        output = self.main(x)
        return output