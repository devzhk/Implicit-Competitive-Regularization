import time

import torch
import torch.autograd as autograd

from .cgd_utils import conjugate_gradient, Hvp_vec, zero_grad


class BCGD(object):
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

            A_x = I + lr ** 2 * h_xy * h_yx
            A_y = I + lr ** 2 * h_yx * h_xy

            cg_x = A_x ** -1 * p_x
            cg_y = A_y ** -1 * p_y

            x = x + lr * cg_x
            y = y - lr * cg_y

            delta x = lr * cg_x
            delta y = - lr * cg_y

            given delta y, solve for delta x:
            delta x = - lr * (grad_x + h_xy-delta_y)
            cg_x = -(grad_x - lr * h_xy-cg_y)

            given delta x, solve for delta y
            delta y = lr * (grad_y + h_yx-delta_x)
            cg_y = -(grad_y + lr * h_yx-cg_x)

            p_x = grad_x - lr * h_xy-v_y
            A_x = I + lr ** 2 * h_xy * h_yx
            cg_x = A_x ** -1 * p_x
            cg_y = -(grad_y + lr * h_yx-cg_x)

            p_y = grad_y + lr * h_yx-v_x
            A_y = I + lr ** 2 * h_yx * h_xy
            cg_y = A_y ** -1 * p_y
            cg_x = -(grad_x - lr * h_xy-cg_y)
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
                            retain_graph=True)  # h_yx * d_x

        p_x = torch.add(grad_x_vec_d, - lr_min * hvp_x_vec).detach_()
        p_y = torch.add(grad_y_vec_d, lr_max * hvp_y_vec).detach_()
        if self.collect_info:
            norm_px = torch.norm(hvp_x_vec, p=2).item()
            norm_py = torch.norm(hvp_y_vec, p=2).item()
            timer = time.time()

        if self.state['solve_x']:
            cg_y, iter_num = conjugate_gradient(grad_x=grad_y_vec, grad_y=grad_x_vec,
                                                x_params=self.min_params,
                                                y_params=self.max_params, b=p_y, x=self.state['old_min'],
                                                nsteps=p_y.shape[0],
                                                lr_x=lr_max, lr_y=lr_min,
                                                device=self.device)
            hcg = Hvp_vec(grad_y_vec, self.max_params, cg_y.detach_()).detach_()
            cg_x = torch.add(grad_x_vec_d, - lr_min * hcg)
        else:
            cg_x, iter_num = conjugate_gradient(grad_x=grad_x_vec, grad_y=grad_y_vec,
                                                x_params=self.max_params,
                                                y_params=self.min_params, b=p_x, x=self.state['old_max'],
                                                nsteps=p_x.shape[0],
                                                lr_x=lr_max, lr_y=lr_min, device=self.device)
            hcg = Hvp_vec(grad_x_vec, self.min_params, cg_x.detach_()).detach_()
            cg_y = torch.add(grad_y_vec_d, lr_max * hcg)
        self.state.update({'old_max': cg_x, 'old_min': cg_y})

        if self.collect_info:
            timer = time.time() - timer
            self.info.update({'time': timer, 'iter_num': iter_num,
                              'hvp_x': norm_px, 'hvp_y': norm_py})

        momentum = self.state['momentum']
        exp_avg_max, exp_avg_min = self.state['exp_avg_max'], self.state['exp_avg_min']
        if momentum != 0:  # TODO test this code: not sure about exp_avg_* initial shape
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

        if self.collect_info:
            norm_gx = torch.norm(grad_x_vec, p=2).item()
            norm_gy = torch.norm(grad_y_vec, p=2).item()
            norm_cgx = torch.norm(cg_x, p=2).item()
            norm_cgy = torch.norm(cg_y, p=2).item()
            self.info.update({'grad_x': norm_gx, 'grad_y': norm_gy,
                              'cg_x': norm_cgx, 'cg_y': norm_cgy})
        self.state['solve_x'] = False if self.state['solve_x'] else True

    def step2(self, loss):
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
                            retain_graph=True)  # h_yx * d_x

        p_x = torch.add(grad_x_vec_d, - lr_min * hvp_x_vec).detach_()
        p_y = torch.add(grad_y_vec_d, lr_max * hvp_y_vec).detach_()
        if self.collect_info:
            norm_px = torch.norm(hvp_x_vec, p=2).item()
            norm_py = torch.norm(hvp_y_vec, p=2).item()
            timer = time.time()
        cg_y, iter_y = conjugate_gradient(grad_x=grad_y_vec, grad_y=grad_x_vec,
                                          x_params=self.min_params,
                                          y_params=self.max_params, b=p_y, x=self.state['old_min'],
                                          nsteps=p_y.shape[0],
                                          lr_x=lr_max, lr_y=lr_min,
                                          device=self.device)
        cg_x, iter_x = conjugate_gradient(grad_x=grad_x_vec, grad_y=grad_y_vec,
                                          x_params=self.max_params,
                                          y_params=self.min_params, b=p_x, x=self.state['old_max'],
                                          nsteps=p_x.shape[0],
                                          lr_x=lr_max, lr_y=lr_min, device=self.device)
        iter_num = iter_x + iter_y
        self.state.update({'old_max': cg_x, 'old_min': cg_y})
        if self.collect_info:
            timer = time.time() - timer
            self.info.update({'time': timer, 'iter_num': iter_num,
                              'hvp_x': norm_px, 'hvp_y': norm_py})

        momentum = self.state['momentum']
        exp_avg_max, exp_avg_min = self.state['exp_avg_max'], self.state['exp_avg_min']
        if momentum != 0:  # TODO test this code: not sure about exp_avg_* initial shape
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

        if self.collect_info:
            norm_gx = torch.norm(grad_x_vec, p=2).item()
            norm_gy = torch.norm(grad_y_vec, p=2).item()
            norm_cgx = torch.norm(cg_x, p=2).item()
            norm_cgy = torch.norm(cg_y, p=2).item()
            self.info.update({'grad_x': norm_gx, 'grad_y': norm_gy,
                              'cg_x': norm_cgx, 'cg_y': norm_cgy})


class BCGD2(object):
    def __init__(self, max_params, min_params,
                 lr_max=1e-3, lr_min=1e-3, device=torch.device('cpu'),
                 update_max=False, collect_info=True):
        self.max_params = list(max_params)
        self.min_params = list(min_params)
        self.device = device
        self.collect_info = collect_info
        self.state = {'lr_max': lr_max, 'lr_min': lr_min,
                      'update_max': update_max, 'old': None}
        self.info = {'grad_x': None, 'grad_y': None,
                     'update': None, 'iter_num': 0}

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
        lr_max = self.state['lr_max']
        lr_min = self.state['lr_min']
        grad_x = autograd.grad(loss, self.max_params, create_graph=True, retain_graph=True)
        grad_x_vec = torch.cat([g.contiguous().view(-1) for g in grad_x])
        grad_y = autograd.grad(loss, self.min_params, create_graph=True, retain_graph=True)
        grad_y_vec = torch.cat([g.contiguous().view(-1) for g in grad_y])
        grad_x_vec_d = grad_x_vec.clone().detach()
        grad_y_vec_d = grad_y_vec.clone().detach()

        if self.state['update_max']:
            hvp_x_vec = Hvp_vec(grad_y_vec, self.max_params, grad_y_vec_d,
                                retain_graph=True)  # h_xy * d_y
            p_x = torch.add(grad_x_vec_d, - lr_min * hvp_x_vec).detach_()
            cg, iter_num = conjugate_gradient(grad_x=grad_x_vec, grad_y=grad_y_vec,
                                              x_params=self.max_params,
                                              y_params=self.min_params, b=p_x, x=self.state['old'],
                                              nsteps=p_x.shape[0],
                                              lr_x=lr_max, lr_y=lr_min, device=self.device)
            cg.detach_()
            index = 0
            for p in self.max_params:
                p.data.add_(lr_max * cg[index: index + p.numel()].reshape(p.shape))
                index += p.numel()
            assert index == cg.numel(), 'Maximizer CG size mismatch'
            self.state.update({'old': cg})
        else:
            hvp_y_vec = Hvp_vec(grad_x_vec, self.min_params, grad_x_vec_d,
                                retain_graph=True)  # h_yx * d_x
            p_y = torch.add(grad_y_vec_d, lr_max * hvp_y_vec).detach_()
            cg, iter_num = conjugate_gradient(grad_x=grad_y_vec, grad_y=grad_x_vec,
                                              x_params=self.min_params,
                                              y_params=self.max_params, b=p_y, x=self.state['old'],
                                              nsteps=p_y.shape[0],
                                              lr_x=lr_max, lr_y=lr_min,
                                              device=self.device)
            cg.detach_()
            index = 0
            for p in self.min_params:
                p.data.add_(- lr_min * cg[index: index + p.numel()].reshape(p.shape))
                index += p.numel()
            assert index == cg.numel(), 'Minimizer CG size mismatch'
            self.state.update({'old': cg})
        if self.collect_info:
            norm_gx = torch.norm(grad_x_vec, p=2).item()
            norm_gy = torch.norm(grad_y_vec, p=2).item()
            norm_cg = torch.norm(cg, p=2).item()
            self.info.update({'grad_x': norm_gx, 'grad_y': norm_gy,
                              'update': norm_cg, 'iter_num': iter_num})
