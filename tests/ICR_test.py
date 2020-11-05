import unittest
import torch
import torch.nn as nn
from optims import ICR, BCGD


lr = 1
cgdType = 'BCGD'
device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'


class NetD(nn.Module):
    def __init__(self):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
        """
        super(NetD, self).__init__()
        self.net = nn.Linear(2, 1)
        self.weight_init()

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return self.net(x)

    def weight_init(self):
        """
        Initialize the weights.

        Args:
            self: (todo): write your description
        """
        self.net.weight.data = torch.Tensor([[1.0, 2.0]])
        self.net.bias.data = torch.Tensor([-1.0])


class NetG(nn.Module):
    def __init__(self):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
        """
        super(NetG, self).__init__()
        self.net = nn.Linear(1, 2)
        self.weight_init()

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return self.net(x)

    def weight_init(self):
        """
        Initialize the weights.

        Args:
            self: (todo): write your description
        """
        self.net.weight.data = torch.Tensor([[3.0], [-1.0]])
        self.net.bias.data = torch.Tensor([-4.0, 3.0])


def print_weight(D, G):
    """
    Print weight of a network.

    Args:
        D: (int): write your description
        G: (int): write your description
    """
    print('===discriminator===')
    print(D.net.weight.data)
    print(D.net.bias.data)
    print('===generator===')
    print(G.net.weight.data)
    print(G.net.bias.data)


class TestICR(unittest.TestCase):
    def setUp(self):
        """
        Set the orientation.

        Args:
            self: (todo): write your description
        """
        self.D_1 = NetD().to(device)
        self.G_1 = NetG().to(device)

        z = torch.tensor([2.0], device=device)
        loss1 = self.D_1(self.G_1(z))
        bcgd = BCGD(max_params=self.G_1.parameters(), min_params=self.D_1.parameters(),
                    lr_min=lr, lr_max=lr, device=device, solve_x=True)
        bcgd.step(loss1)
        print('BCGD step')
        self.D_2 = NetD().to(device)
        self.G_2 = NetG().to(device)
        print_weight(D=self.D_2, G=self.G_2)
        loss2 = self.D_2(self.G_2(z))
        icr = ICR(max_params=self.G_2.parameters(), min_params=self.D_2.parameters(),
                  lr=lr, alpha=1.0, device=device)
        icr.step(loss2)
        print('ICR step')

    def testDw_equal(self):
        """
        This function is called d_1 and d_2 are equal.

        Args:
            self: (todo): write your description
        """
        self.assertEqual(self.D_1.net.weight.data.tolist(), self.D_2.net.weight.data.tolist())

    def testDb_equal(self):
        """
        Test if the network is equal

        Args:
            self: (todo): write your description
        """
        self.assertEqual(self.D_1.net.bias.data.tolist(), self.D_2.net.bias.data.tolist())

    def testGw_equal(self):
        """
        Test if g_2 ( gw is equal to g_1.

        Args:
            self: (todo): write your description
        """
        self.assertEqual(self.G_1.net.weight.data.tolist(), self.G_2.net.weight.data.tolist())

    def testGb_equal(self):
        """
        Test if g_equal is equal to g_2.

        Args:
            self: (todo): write your description
        """
        self.assertEqual(self.G_1.net.bias.data.tolist(), self.G_2.net.bias.data.tolist())

    def tearDown(self):
        """
        Tear tear interpolation.

        Args:
            self: (todo): write your description
        """
        print('Model weights after one step:')
        print_weight(D=self.D_1, G=self.G_1)
        print('=====BCGD=====')
        print_weight(D=self.D_2, G=self.G_2)
        print('=====ICR=====')


if __name__ == '__main__':
    # unittest.main()
    factor = 0.01
    A_g = torch.tensor([[1.0 + 4.0 * factor, 0.0, 2.0 * factor, 0],
                        [0.0, 1.0 + 4.0 * factor, 0.0, 2.0 * factor],
                        [2.0 * factor, 0.0, 1.0 + factor, 0.0],
                        [0.0, 2.0 * factor, 0.0, 1.0 + factor]])
    A_d = torch.tensor([[1.0 + 5.0 * factor, 0.0, 0.0],
                        [0.0, 1.0 + 5.0 * factor, 0.0],
                        [0.0, 0.0, 1.0]])
