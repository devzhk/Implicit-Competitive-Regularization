import torch
from .cgd_utils import zero_grad
import matplotlib.pyplot as plt
import torch.autograd as autograd

plt.switch_backend('agg')
import numpy as np
import os
from optims.cgd_utils import conjugate_gradient, Hvp_vec, zero_grad
from scipy.sparse.linalg import cg


class ICR(object):
    def __init__(self, max_params, min_params,
                 lr=1e-3, momentum=0, alpha=1.0,
                 device=torch.device('cpu'), collect_info=True):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            max_params: (int): write your description
            min_params: (dict): write your description
            lr: (float): write your description
            momentum: (array): write your description
            alpha: (float): write your description
            device: (todo): write your description
            torch: (todo): write your description
            device: (todo): write your description
            collect_info: (todo): write your description
        """
        self.max_params = list(max_params)
        self.min_params = list(min_params)
        self.state = {'lr': lr, 'momentum': momentum, 'alpha': alpha,
                      'step': 0, 'old_max': None, 'old_min': None,  # start point of CG
                      'exp_avg_max': 0.0, 'exp_avg_min': 0.0}  # save last update
        self.info = {'grad_x': None, 'grad_y': None,
                     'hvp_x': None, 'hvp_y': None,
                     'cg_x': None, 'cg_y': None,
                     'time': 0, 'iter_num': 0}
        self.device = device
        self.collect_info = collect_info

    def zero_grad(self):
        """
        Calculate the gradient

        Args:
            self: (todo): write your description
        """
        zero_grad(self.max_params)
        zero_grad(self.min_params)

    def state_dict(self):
        """
        : return a dictionary with the state

        Args:
            self: (todo): write your description
        """
        return self.state

    def load_state_dict(self, state_dict):
        """
        Loads the state dictionary from a dictionary.

        Args:
            self: (todo): write your description
            state_dict: (dict): write your description
        """
        self.state.update(state_dict)

    def set_state(self, new_state):
        """
        Set the state of a new layer.

        Args:
            self: (todo): write your description
            new_state: (int): write your description
        """
        self.state.update(new_state)
        print('Current state: {}'.format(new_state))

    def get_info(self):
        """
        Get info about the server

        Args:
            self: (todo): write your description
        """
        if self.info['grad_x'] is None:
            print('Warning! No update information stored. Set collect_info=True before call this method')
        return self.info

    def step(self, loss):
        """
        Perform a single optimization step.

        Args:
            self: (todo): write your description
            loss: (todo): write your description
        """
        lr = self.state['lr']
        alpha = self.state['alpha']
        time_step = self.state['step'] + 1
        self.state['step'] = time_step

        grad_x = autograd.grad(loss, self.max_params, create_graph=True, retain_graph=True)
        grad_x_vec = torch.cat([g.contiguous().view(-1) for g in grad_x])
        grad_y = autograd.grad(loss, self.min_params, create_graph=True, retain_graph=True)
        grad_y_vec = torch.cat([g.contiguous().view(-1) for g in grad_y])
        grad_x_vec_d = grad_x_vec.clone().detach()
        grad_y_vec_d = grad_y_vec.clone().detach()
        hvp_x_vec = Hvp_vec(grad_y_vec, self.max_params, grad_y_vec_d,
                            retain_graph=True)  # h_xy * d_y
        hvp_y_vec = Hvp_vec(grad_x_vec, self.min_params, grad_x_vec_d,
                            retain_graph=True)  # h_yx * d_x
        p_x = torch.add(grad_x_vec_d, - lr * alpha * hvp_x_vec).detach_()
        p_y = torch.add(grad_y_vec_d, lr * alpha * hvp_y_vec).detach_()

        cg_x, x_i = conjugate_gradient(grad_x=grad_x_vec, grad_y=grad_y_vec,
                                       x_params=self.max_params, y_params=self.min_params,
                                       b=p_x, x=self.state['old_max'],
                                       nsteps=p_x.shape[0], lr_x=lr, lr_y=alpha * lr,
                                       device=self.device)
        cg_x.detach_()
        cg_y, y_i = conjugate_gradient(grad_x=grad_y_vec, grad_y=grad_x_vec,
                                       x_params=self.min_params, y_params=self.max_params,
                                       b=p_y, x=self.state['old_min'],
                                       nsteps=p_y.shape[0], lr_x=lr, lr_y=alpha * lr,
                                       device=self.device)
        cg_y.detach_()

        # path = 'figs/ICR/'
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # plot_cg(tensorList=g_array, name='%sgenerator_%d' % (path, time_step))
        # plot_cg(tensorList=d_array, name='%sdiscriminator_%d' % (path, time_step))
        self.state.update({'old_max': cg_x, 'old_min': cg_y})

        index = 0
        for p in self.max_params:
            p.data.add_(lr * cg_x[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != cg_x.numel():
            raise ValueError('CG size mismatch')
        index = 0
        for p in self.min_params:
            p.data.add_(- lr * cg_y[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != cg_y.numel():
            raise ValueError('CG size mismatch')


class testBCGD(object):
    def __init__(self, max_params, min_params,
                 lr_max=1e-3, lr_min=1e-3,
                 device=torch.device('cpu')):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            max_params: (int): write your description
            min_params: (dict): write your description
            lr_max: (float): write your description
            lr_min: (float): write your description
            device: (todo): write your description
            torch: (todo): write your description
            device: (todo): write your description
        """
        self.max_params = list(max_params)
        self.min_params = list(min_params)
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.device = device
        self.count = 0
        self.state = {'old_g': None, 'old_d': None}

    def zero_grad(self):
        """
        Calculate the gradient

        Args:
            self: (todo): write your description
        """
        zero_grad(self.max_params)
        zero_grad(self.min_params)

    def step(self, loss):
        """
        Perform one step.

        Args:
            self: (todo): write your description
            loss: (todo): write your description
        """
        g_param = torch.cat([p.contiguous().view(-1) for p in self.max_params])
        d_param = torch.cat([p.contiguous().view(-1) for p in self.min_params])
        grad_g = torch.tensor([2 * d_param[0].data, 2 * d_param[1].data,
                               d_param[0].data, d_param[1].data])
        grad_d = torch.tensor([2 * g_param[0].data - 3.0 + g_param[2].data,
                               2 * g_param[1].data - 4.0 + g_param[3].data, 0.0])

        hvp_g = torch.tensor([4 * g_param[0].data - 6.0 + 2 * g_param[2].data,
                              4 * g_param[1].data - 8.0 + 2 * g_param[3].data,
                              2 * g_param[0].data - 3.0 + g_param[2].data,
                              2 * g_param[1].data - 4.0 + g_param[3].data])
        hvp_d = torch.tensor([5 * d_param[0].data, 5 * d_param[1].data, 0.0])
        p_g = torch.add(grad_g, - self.lr_min * hvp_g)
        p_d = torch.add(grad_d, self.lr_max * hvp_d)
        factor = self.lr_max * self.lr_min
        A_g = torch.tensor([[1.0 + 4.0 * factor, 0.0, 2.0 * factor, 0],
                            [0.0, 1.0 + 4.0 * factor, 0.0, 2.0 * factor],
                            [2.0 * factor, 0.0, 1.0 + factor, 0.0],
                            [0.0, 2.0 * factor, 0.0, 1.0 + factor]])
        A_d = torch.tensor([[1.0 + 5.0 * factor, 0.0, 0.0],
                            [0.0, 1.0 + 5.0 * factor, 0.0],
                            [0.0, 0.0, 1.0]])
        cg_g, iter_g, g_array = inverse(A=A_g, b=p_g, nsteps=p_g.shape[0])
        cg_d, iter_d, d_array = inverse(A=A_d, b=p_d, nsteps=p_d.shape[0])
        self.state.update({'old_g': cg_g, 'old_d': cg_d})
        # path = 'figs/testBCGD/'
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # plot_cg(tensorList=g_array, name='%sgenerator_%d' % (path, self.count))
        # plot_cg(tensorList=d_array, name='%sdiscriminator_%d' % (path, self.count))
        # cg_g, iter_g = inverse_gt(A=A_g, b=p_g)
        # cg_d, iter_d = inverse_gt(A=A_d, b=p_d)
        index = 0
        for p in self.max_params:
            p.data.add_(self.lr_max * cg_g[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        assert index == cg_g.numel(), 'Maximizer CG size mismatch'
        index = 0
        for p in self.min_params:
            p.data.add_(- self.lr_min * cg_d[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        assert index == cg_d.numel(), 'Minimizer CG size mismatch'
        self.count += 1


class testLCGD(object):
    def __init__(self, max_params, min_params,
                 lr_max=1e-3, lr_min=1e-3,
                 device=torch.device('cpu')):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            max_params: (int): write your description
            min_params: (dict): write your description
            lr_max: (float): write your description
            lr_min: (float): write your description
            device: (todo): write your description
            torch: (todo): write your description
            device: (todo): write your description
        """
        self.max_params = list(max_params)
        self.min_params = list(min_params)
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.device = device

    def zero_grad(self):
        """
        Calculate the gradient

        Args:
            self: (todo): write your description
        """
        zero_grad(self.max_params)
        zero_grad(self.min_params)

    def step(self, loss):
        """
        Perform one step.

        Args:
            self: (todo): write your description
            loss: (todo): write your description
        """
        g_param = torch.cat([p.contiguous().view(-1) for p in self.max_params])
        d_param = torch.cat([p.contiguous().view(-1) for p in self.min_params])
        grad_g = torch.tensor([2 * d_param[0].data, 2 * d_param[1].data,
                               d_param[0].data, d_param[1].data])
        grad_d = torch.tensor([2 * g_param[0].data - 3.0 + g_param[2].data,
                               2 * g_param[1].data - 4.0 + g_param[3].data, 0.0])

        hvp_g = torch.tensor([4 * g_param[0].data - 6.0 + 2 * g_param[2].data,
                              4 * g_param[1].data - 8.0 + 2 * g_param[3].data,
                              2 * g_param[0].data - 3.0 + g_param[2].data,
                              2 * g_param[1].data - 4.0 + g_param[3].data])
        hvp_d = torch.tensor([5 * d_param[0].data, 5 * d_param[1].data, 0.0])
        cg_g = torch.add(grad_g, - self.lr_min * hvp_g)
        cg_d = torch.add(grad_d, self.lr_max * hvp_d)
        index = 0
        for p in self.max_params:
            p.data.add_(self.lr_max * cg_g[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        assert index == cg_g.numel(), 'Maximizer CG size mismatch'
        index = 0
        for p in self.min_params:
            p.data.add_(- self.lr_min * cg_d[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        assert index == cg_d.numel(), 'Minimizer CG size mismatch'


def inverse(A, b, x=None, nsteps=10,
            tol=1e-12, atol=1e-20):
    """
    return A ** -1 * b
    : param tol: relative tolerance
    : param atol: absolute tolerance
    """
    if x is None:
        x = torch.zeros_like(b)
    r = b.clone().detach()
    p = r.clone().detach()
    rdotr = torch.dot(r, r)
    residual_tol = tol * rdotr
    pArray = []
    for i in range(nsteps):
        pArray.append(p)

        Avp_ = torch.matmul(A, p)
        pArray.append(Avp_)

        alpha = rdotr / torch.dot(p, Avp_)
        x.data.add_(alpha * p)
        pArray.append(x.data)

        r.data.add_(- alpha * Avp_)
        new_rdotr = torch.dot(r, r)
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
        if rdotr < residual_tol or rdotr < atol:
            break
    return x, i + 1, pArray


def inverse_gt(A, b):
    """
    Inverse of the matrix b.

    Args:
        A: (int): write your description
        b: (int): write your description
    """
    return torch.matmul(torch.inverse(A), b), 0


def plot_cg(tensorList, name):
    """
    Plot tensor of tensor.

    Args:
        tensorList: (list): write your description
        name: (str): write your description
    """
    array = []
    for p in tensorList:
        array.append(p.tolist())
    array = np.array(array).transpose()
    np.save('%s.npy' % name, array)
    index = np.arange(array.shape[1])
    fig, axs = plt.subplots(array.shape[0])
    for i, a in enumerate(array):
        axs[i].plot(index, a)
        axs[i].set_title('Feature %d' % i)
    plt.savefig('%s.png' % name)
