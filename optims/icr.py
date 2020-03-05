import time
import torch
import torch.autograd as autograd

from .cgd_utils import conjugate_gradient, Hvp_vec, zero_grad


class ICR(object):
    def __init__(self, max_params, min_params,
                 lr=1e-3, momentum=0, alpha=2.0,
                 device=torch.device('cpu'), collect_info=True):
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
        zero_grad(self.max_params)
        zero_grad(self.min_params)

    def state_dict(self):
        return self.state

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)

    def set_state(self, new_state):
        self.state.update(new_state)
        print('Current state: {}'.format(new_state))

    def get_info(self):
        if self.info['grad_x'] is None:
            print('Warning! No update information stored. Set collect_info=True before call this method')
        return self.info

    def step(self, loss):
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

        if self.collect_info:
            norm_px = torch.norm(hvp_x_vec, p=2).detach_()
            norm_py = torch.norm(hvp_y_vec, p=2).detach_()
            timer = time.time()

        cg_x, x_iter_num = conjugate_gradient(grad_x=grad_x_vec, grad_y=grad_y_vec,
                                              x_params=self.max_params, y_params=self.min_params,
                                              b=p_x, x=self.state['old_max'],
                                              nsteps=p_x.shape[0], lr_x=lr, lr_y=alpha * lr,
                                              device=self.device)
        cg_y, y_iter_num = conjugate_gradient(grad_x=grad_y_vec, grad_y=grad_x_vec,
                                              x_params=self.min_params, y_params=self.max_params,
                                              b=p_y, x=self.state['old_min'],
                                              nsteps=p_y.shape[0], lr_x=lr, lr_y=alpha * lr,
                                              device=self.device)
        iter_num = x_iter_num + y_iter_num
        cg_x.detach_()
        cg_y.detach_()
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

        if self.collect_info:
            norm_gx = torch.norm(grad_x_vec_d, p=2)
            norm_gy = torch.norm(grad_y_vec_d, p=2)
            norm_cgx = torch.norm(cg_x, p=2)
            norm_cgy = torch.norm(cg_y, p=2)
            timer = time.time() - timer
            self.info.update({'grad_x': norm_gx, 'grad_y': norm_gy,
                              'hvp_x': norm_px, 'hvp_y': norm_py,
                              'cg_x': norm_cgx, 'cg_y': norm_cgy,
                              'time': timer, 'iter_num': iter_num})










