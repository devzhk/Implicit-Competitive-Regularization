import time

import torch
import torch.autograd as autograd

from .cgd_utils import conjugate_gradient, Hvp_vec, zero_grad


class LCGD(object):
    def __init__(self, max_params, min_params,
                 lr_max=1e-3, lr_min=1e-3,
                 momentum=0.0, device=torch.device('cpu'),
                 solve_x=False, collect_info=True):
        self.max_params = list(max_params)
        self.min_params = list(min_params)
        self.state = {'lr_max': lr_max, 'lr_min': lr_min,
                      'momentum': momentum, 'solve_x': solve_x,
                      'step': 0, 'old_max': None, 'old_min': None,  # start point of CG
                      'exp_avg_max': 0.0, 'exp_avg_min': 0.0}  # save last update
        self.info = {'grad_x': None, 'grad_y': None,
                     'hvp_x': None, 'hvp_y': None,
                     'cg_x': None, 'cg_y': None,
                     'time': 0, 'iter_num': 0}
        self.device = device
        self.collect_info = collect_info

    def zero_grad(self):
        zero_grad(self.max_params)
        zero_grad(self.min_params)

    def get_info(self):
        if self.info['grad_x'] is None:
            print('Warning! No update information stored. Set collect_info=True before call this method')
        return self.info

    def state_dict(self):
        return self.state

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)
        print('Load state: {}'.format(state_dict))

    def set_lr(self, lr_max, lr_min):
        self.state.update({'lr_max': lr_max, 'lr_min': lr_min})
        print('Maximizing side learning rate: {:.4f}\n '
              'Minimizing side learning rate: {:.4f}'.format(lr_max, lr_min))

    def step(self, loss):
        """
            update rules:
            p_x = grad_x - lr * h_xy-v_y
            p_y = grad_y + lr * h_yx-v_x

            x = x + lr * p_x
            y = y - lr * p_y
        """
        lr_max = self.state['lr_max']
        lr_min = self.state['lr_min']
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
                            retain_graph=False)  # h_yx * d_x

        cg_x = torch.add(grad_x_vec_d, - lr_min * hvp_x_vec).detach_()
        cg_y = torch.add(grad_y_vec_d, lr_max * hvp_y_vec).detach_()

        momentum = self.state['momentum']
        exp_avg_max, exp_avg_min = self.state['exp_avg_max'], self.state['exp_avg_min']
        if momentum != 0:
            bias_correction = 1 - momentum ** time_step
            lr_max /= bias_correction
            lr_min /= bias_correction
            cg_x = exp_avg_max.mul(momentum) + cg_x.mul(1 - momentum)
            cg_y = exp_avg_min.mul(momentum) + cg_y.mul(1 - momentum)
        index = 0
        for p in self.max_params:
            p.data.add_(lr_max * cg_x[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        assert index == cg_x.numel(), 'Maximizer CG size mismatch'
        index = 0
        for p in self.min_params:
            p.data.add_(- lr_min * cg_y[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        assert index == cg_y.numel(), 'Minimizer CG size mismatch'
