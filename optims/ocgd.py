import time
import math
import torch
import torch.autograd as autograd

from .cgd_utils import zero_grad, general_conjugate_gradient, Hvp_vec


class OCGD(object):
    def __init__(self, max_params, min_params,
                 lr_max=1e-3, lr_min=1e-3,
                 eps=1e-5, beta=0.99,
                 device=torch.device('cpu'),
                 udpate_min=False, collect_info=True):
        """
        Implements.

        Args:
            self: (todo): write your description
            max_params: (int): write your description
            min_params: (dict): write your description
            lr_max: (float): write your description
            lr_min: (float): write your description
            eps: (float): write your description
            beta: (float): write your description
            device: (todo): write your description
            torch: (todo): write your description
            device: (todo): write your description
            udpate_min: (str): write your description
            collect_info: (todo): write your description
        """
        self.max_params = list(max_params)
        self.min_params = list(min_params)
        self.state = {'lr_max': lr_max, 'lr_min': lr_min,
                      'eps': eps, 'beta': beta, 'step': 0,
                      'old_max': None, 'old_min': None,  # start point of CG
                      'sq_exp_avg_max': None, 'sq_exp_avg_min': None}  # save last update
        self.info = {'grad_x': None, 'grad_y': None,
                     'hvp_x': None, 'hvp_y': None,
                     'cg_x': None, 'cg_y': None,
                     'time': 0, 'iter_num': 0}
        self.device = device
        self.collect_info = collect_info
        self.update_min = udpate_min

    def zero_grad(self):
        """
        Calculate the gradient

        Args:
            self: (todo): write your description
        """
        zero_grad(self.max_params)
        zero_grad(self.min_params)

    def getinfo(self):
        """
        Returns the info about the server

        Args:
            self: (todo): write your description
        """
        if self.info['grad_x'] is None:
            print('Warning! No update information stored. Set collect_info=True before call this method')
        return self.info

    def state_dict(self):
        """
        : return a dictionary with the state

        Args:
            self: (todo): write your description
        """
        return self.state

    def load_state_dict(self, state_dict):
        """
        Loads the state from a dictionary.

        Args:
            self: (todo): write your description
            state_dict: (dict): write your description
        """
        self.state.update(state_dict)
        print('Load state: {}'.format(state_dict))

    def set_lr(self, lr_max, lr_min):
        """
        Sets the learning rate.

        Args:
            self: (todo): write your description
            lr_max: (int): write your description
            lr_min: (todo): write your description
        """
        self.state.update({'lr_max': lr_max, 'lr_min': lr_min})
        print('Maximizing side learning rate: {:.4f}\n '
              'Minimizing side learning rate: {:.4f}'.format(lr_max, lr_min))

    def step(self, loss):
        """
        Perform a single optimization step.

        Args:
            self: (todo): write your description
            loss: (todo): write your description
        """
        lr_max = self.state['lr_max']
        lr_min = self.state['lr_min']
        beta = self.state['beta']
        eps = self.state['eps']
        time_step = self.state['step'] + 1
        self.state['step'] = time_step

        grad_x = autograd.grad(loss, self.max_params, create_graph=True, retain_graph=True)
        grad_x_vec = torch.cat([g.contiguous().view(-1) for g in grad_x])
        grad_y = autograd.grad(loss, self.min_params, create_graph=True, retain_graph=True)
        grad_y_vec = torch.cat([g.contiguous().view(-1) for g in grad_y])
        grad_x_vec_d = grad_x_vec.clone().detach()
        grad_y_vec_d = grad_y_vec.clone().detach()

        sq_avg_x = self.state['sq_exp_avg_max']
        sq_avg_y = self.state['sq_exp_avg_min']
        sq_avg_x = torch.zeros_like(grad_x_vec_d, requires_grad=False) if sq_avg_x is None else sq_avg_x
        sq_avg_y = torch.zeros_like(grad_y_vec_d, requires_grad=False) if sq_avg_y is None else sq_avg_y
        sq_avg_x.mul_(beta).addcmul_(1 - beta, grad_x_vec_d, grad_x_vec_d)
        sq_avg_y.mul_(beta).addcmul_(1 - beta, grad_y_vec_d, grad_y_vec_d)

        bias_correction = 1 - beta ** time_step
        lr_max = math.sqrt(bias_correction) * lr_max / sq_avg_x.sqrt().add(eps)
        lr_min = math.sqrt(bias_correction) * lr_min / sq_avg_y.sqrt().add(eps)

        scaled_grad_x = torch.mul(lr_max, grad_x_vec_d)
        scaled_grad_y = torch.mul(lr_min, grad_y_vec_d)
        hvp_x_vec = Hvp_vec(grad_y_vec, self.max_params, scaled_grad_y,
                            retain_graph=True)  # h_xy * d_y
        hvp_y_vec = Hvp_vec(grad_x_vec, self.min_params, scaled_grad_x,
                            retain_graph=True)  # h_yx * d_x
        p_x = torch.add(grad_x_vec_d, - hvp_x_vec)
        p_y = torch.add(grad_y_vec_d, hvp_y_vec)

        if self.update_min:
            p_y.mul_(lr_min.sqrt())
            cg_y, iter_num = general_conjugate_gradient(grad_x=grad_y_vec, grad_y=grad_x_vec,
                                                        x_params=self.min_params, y_params=self.max_params,
                                                        b=p_y, x=self.state['old_min'],
                                                        lr_x=lr_min, lr_y=lr_max, device=self.device)
            old_min = cg_y.detach_()
            self.state.update({'old_min': old_min})
            min_update = cg_y.mul(- lr_min.sqrt())
            index = 0
            for p in self.min_params:
                p.data.add_(min_update[index: index + p.numel()].reshape(p.shape))
                index += p.numel()
            assert index == min_update.numel(), 'Minimizer CG size mismatch'
        else:
            p_x.mul_(lr_max.sqrt())
            cg_x, iter_num = general_conjugate_gradient(grad_x=grad_x_vec, grad_y=grad_y_vec,
                                                        x_params=self.max_params, y_params=self.min_params,
                                                        b=p_x, x=self.state['old_max'],
                                                        lr_x=lr_max, lr_y=lr_min, device=self.device)
            old_max = cg_x.detach_()
            self.state.update({'old_min': old_max})
            max_update = cg_x.mul(lr_max.sqrt())
            index = 0
            for p in self.max_params:
                p.data.add_(max_update[index: index + p.numel()].reshape(p.shape))
                index += p.numel()
            assert index == max_update.numel(), 'Maximizer CG size mismatch'

        self.state.update({'sq_exp_avg_max': sq_avg_x, 'sq_exp_avg_min': sq_avg_y})






