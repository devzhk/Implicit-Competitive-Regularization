import math
import time

import torch
import torch.autograd as autograd

from .cgd_utils import conjugate_gradient, general_conjugate_gradient, Hvp_vec, zero_grad


class BCGD(object):
    def __init__(self, max_params, min_params,
                 lr=1e-3, momentum=0, device=torch.device('cpu'),
                 solve_x=False, collect_info=True):
        self.max_params = list(max_params)
        self.min_params = list(min_params)
        self.state = {'lr': lr, 'momentum': momentum, 'solve_x': solve_x,
                      'step': 0,'old_max': None, 'old_min': None, # start point of CG
                      'exp_avg_max': 0.0, 'exp_avg_min': 0.0} # save last update
        # self.lr = lr
        # self.momentum = momentum
        self.device = device
        # self.solve_x = solve_x
        self.collect_info = collect_info

        # self.old_x = None
        # self.old_y = None

    def zero_grad(self):
        zero_grad(self.max_params)
        zero_grad(self.min_params)

    def getinfo(self):
        if self.collect_info:
            return self.norm_gx, self.norm_gy, self.norm_px, self.norm_py, self.norm_cgx, self.norm_cgy, \
                   self.timer, self.iter_num
        else:
            raise ValueError(
                'No update information stored. Set collect_info=True before call this method')

    def step(self, loss):
        lr = self.state['lr']
        solve_x = self.state['solve_x']
        time_step = self.state['step'] + 1
        self.state['step'] = time_step

        grad_x = autograd.grad(loss, self.max_params, create_graph=True, retain_graph=True)
        grad_x_vec = torch.cat([g.contiguous().view(-1) for g in grad_x])
        grad_y = autograd.grad(loss, self.min_params, create_graph=True, retain_graph=True)
        grad_y_vec = torch.cat([g.contiguous().view(-1) for g in grad_y])

        hvp_x_vec = Hvp_vec(grad_y_vec, self.max_params, grad_y_vec,
                            retain_graph=True)  # h_xy * d_y
        hvp_y_vec = Hvp_vec(grad_x_vec, self.min_params, grad_x_vec,
                            retain_graph=True)  # h_yx * d_x

        p_x = torch.add(grad_x_vec, - lr * hvp_x_vec)
        p_y = torch.add(grad_y_vec, lr * hvp_y_vec)
        if self.collect_info:
            self.norm_px = torch.norm(p_x, p=2)
            self.norm_py = torch.norm(p_y, p=2)
            self.timer = time.time()

        if solve_x:
            cg_y, self.iter_num = conjugate_gradient(grad_x=grad_y_vec, grad_y=grad_x_vec,
                                                     x_params=self.min_params,
                                                     y_params=self.max_params, b=p_y, x=self.state['old_min'],
                                                     nsteps=p_y.shape[0] // 1000,
                                                     lr_x=lr, lr_y=lr, device=self.device)
            hcg = Hvp_vec(grad_y_vec, self.max_params, cg_y)
            cg_x = torch.add(grad_x_vec, - lr * hcg)
        else:
            cg_x, self.iter_num = conjugate_gradient(grad_x=grad_x_vec, grad_y=grad_y_vec,
                                                     x_params=self.max_params,
                                                     y_params=self.min_params, b=p_x, x=self.state['old_max'],
                                                     nsteps=p_x.shape[0] // 1000,
                                                     lr_x=lr, lr_y=lr, device=self.device)
            hcg = Hvp_vec(grad_x_vec, self.min_params, cg_x)
            cg_y = torch.add(grad_y_vec, lr * hcg)
        self.state.update({'old_max': cg_x, 'old_min': cg_y})

        if self.collect_info:
            self.timer = time.time() - self.timer

        momentum = self.state['momentum']
        bias_correction = 1 - momentum ** time_step
        lr /= bias_correction
        exp_avg_max, exp_avg_min = self.state['exp_avg_max'], self.state['exp_avg_min']
        if momentum != 0: #TODO test this code: not sure about exp_avg_* initial shape
            cg_x = exp_avg_max.mul(momentum) + cg_x.mul(1 - momentum)
            cg_y = exp_avg_min.mul(momentum) + cg_y.mul(1 - momentum)
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

        if self.collect_info:
            self.norm_gx = torch.norm(grad_x_vec, p=2)
            self.norm_gy = torch.norm(grad_y_vec, p=2)
            self.norm_cgx = torch.norm(cg_x, p=2)
            self.norm_cgy = torch.norm(cg_y, p=2)
        self.state['solve_x'] = False if solve_x else True