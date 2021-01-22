import time
import math
import torch
import torch.autograd as autograd

from .cgd_utils import zero_grad, general_conjugate_gradient, Hvp_vec, gd_solver


class ACGD(object):
    def __init__(self, max_params, min_params,
                 lr_max=1e-3, lr_min=1e-3,
                 eps=1e-5, beta=0.99,
                 tol=1e-12, atol=1e-20,
                 device=torch.device('cpu'),
                 solve_x=False, collect_info=True,
                 solver='cg'):
        self.max_params = list(max_params)
        self.min_params = list(min_params)
        self.state = {'lr_max': lr_max, 'lr_min': lr_min,
                      'eps': eps, 'solve_x': solve_x,
                      'tol': tol, 'atol': atol,
                      'beta': beta, 'step': 0,
                      'old_max': None, 'old_min': None,  # start point of CG
                      'sq_exp_avg_max': None, 'sq_exp_avg_min': None}  # save last update
        self.info = {'grad_x': None, 'grad_y': None,
                     'hvp_x': None, 'hvp_y': None,
                     'cg_x': None, 'cg_y': None,
                     'time': 0, 'iter_num': 0}
        self.device = device
        self.collect_info = collect_info
        self.solver= solver

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
        # print('Maximizing side learning rate: {:.4f}\n '
        #       'Minimizing side learning rate: {:.4f}'.format(lr_max, lr_min))

    def step(self, loss):
        lr_max = self.state['lr_max']
        lr_min = self.state['lr_min']
        beta = self.state['beta']
        eps = self.state['eps']
        tol = self.state['tol']
        atol = self.state['atol']
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

        sq_avg_x = sq_avg_x.to(self.device)
        sq_avg_y = sq_avg_y.to(self.device)

        sq_avg_x.mul_(beta).addcmul_(grad_x_vec_d, grad_x_vec_d, value=1 - beta)
        sq_avg_y.mul_(beta).addcmul_(grad_y_vec_d, grad_y_vec_d, value=1 - beta)

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
        if self.collect_info:
            norm_px = torch.norm(hvp_x_vec, p=2).item()
            norm_py = torch.norm(hvp_y_vec, p=2).item()
            timer = time.time()

        if self.state['solve_x']:
            p_y.mul_(lr_min.sqrt())
            if self.solver == 'cg':
                cg_y, iter_num = general_conjugate_gradient(grad_x=grad_y_vec, grad_y=grad_x_vec,
                                                            x_params=self.min_params, y_params=self.max_params,
                                                            b=p_y, x=self.state['old_min'],
                                                            tol=tol, atol=atol,
                                                            lr_x=lr_min, lr_y=lr_max,
                                                            nsteps=None,
                                                            device=self.device)
            elif self.solver == 'gd':
                cg_y, iter_num = gd_solver(grad_x=grad_y_vec, grad_y=grad_x_vec,
                                           x_params=self.min_params, y_params=self.max_params,
                                           b=p_y, x=self.state['old_min'],
                                           lr_x=lr_min, lr_y=lr_max, device=self.device)
            old_min = cg_y.detach_()
            min_update = cg_y.mul(- lr_min.sqrt())
            hcg = Hvp_vec(grad_y_vec, self.max_params, min_update).detach_()
            hcg.add_(grad_x_vec_d)
            max_update = hcg.mul(lr_max)
            old_max = hcg.mul(lr_max.sqrt())
        else:
            p_x.mul_(lr_max.sqrt())
            if self.solver == 'cg':
                cg_x, iter_num = general_conjugate_gradient(grad_x=grad_x_vec, grad_y=grad_y_vec,
                                                            x_params=self.max_params, y_params=self.min_params,
                                                            b=p_x, x=self.state['old_max'],
                                                            tol=tol, atol=atol,
                                                            lr_x=lr_max, lr_y=lr_min,
                                                            nsteps=1000,
                                                            device=self.device)
            elif self.solver == 'gd':
                cg_x, iter_num = gd_solver(grad_x=grad_x_vec, grad_y=grad_y_vec,
                                           x_params=self.max_params, y_params=self.min_params,
                                           b=p_x, x=self.state['old_max'],
                                           lr_x=lr_max, lr_y=lr_min, device=self.device)
            old_max = cg_x.detach_()
            max_update = cg_x.mul(lr_max.sqrt())
            hcg = Hvp_vec(grad_x_vec, self.min_params, max_update).detach_()
            hcg.add_(grad_y_vec_d)
            min_update = hcg.mul(- lr_min)
            old_min = hcg.mul(lr_min.sqrt())
        self.state.update({'old_max': old_max, 'old_min': old_min,
                           'sq_exp_avg_max': sq_avg_x, 'sq_exp_avg_min': sq_avg_y})

        if self.collect_info:
            timer = time.time() - timer
            self.info.update({'time': timer, 'iter_num': iter_num,
                              'hvp_x': norm_px, 'hvp_y': norm_py})

        index = 0
        for p in self.max_params:
            p.data.add_(max_update[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        assert index == max_update.numel(), 'Maximizer CG size mismatch'

        index = 0
        for p in self.min_params:
            p.data.add_(min_update[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        assert index == min_update.numel(), 'Minimizer CG size mismatch'

        if self.collect_info:
            norm_gx = torch.norm(grad_x_vec, p=2).item()
            norm_gy = torch.norm(grad_y_vec, p=2).item()
            norm_cgx = torch.norm(max_update, p=2).item()
            norm_cgy = torch.norm(min_update, p=2).item()
            self.info.update({'grad_x': norm_gx, 'grad_y': norm_gy,
                              'cg_x': norm_cgx, 'cg_y': norm_cgy})
        self.state['solve_x'] = False if self.state['solve_x'] else True




