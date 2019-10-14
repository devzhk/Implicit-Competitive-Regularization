from cgd_utils import conjugate_gradient, Hvp, Hvp_vec, general_conjugate_gradient, Hvpvec, mgeneral_conjugate_gradient
import torch.autograd as autograd
import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required
import time
import math


def zero_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.detach()
            p.grad.zero_()


def fLCGD(loss, max_params, min_params, lr=1e-2, weight_decay=0):
    '''
    max_x min_y loss
    :param loss: object function
    :param max_params: x
    :param min_params: y
    :param lr: learning rate
    :param weight_decay: weight decay rate recommend value: 0.01, 0.001, 0.0001
    :return: weight value
    x = x + lr * (grad_x - lr * h)
    y = y - lr * (grad_y + lr * h)
    '''
    zero_grad(max_params)
    zero_grad(min_params)

    grad_x = autograd.grad(loss, max_params, create_graph=True, retain_graph=True)
    grad_x_vec = torch.cat([g.contiguous().view(-1) for g in grad_x])
    grad_y = autograd.grad(loss, min_params, create_graph=True, retain_graph=True)
    grad_y_vec = torch.cat([g.contiguous().view(-1) for g in grad_y])
    hvp_x = Hvp(grad_y_vec, max_params, grad_y_vec, retain_graph=True)
    hvp_y = Hvp(grad_x_vec, min_params, grad_x_vec, retain_graph=True)

    for p, grad, h in zip(max_params, grad_x, hvp_x):
        if weight_decay != 0:
            p.data.add_(- weight_decay * p)
        d_p = grad.data
        d_p.add_(-lr * h)
        p.data.add_(lr * d_p)


    for p, grad, h in zip(min_params, grad_y, hvp_y):
        if weight_decay != 0:
            p.data.add_(- weight_decay * p)
        d_p = grad.data
        d_p.add_(lr * h)
        p.data.add_(- lr * d_p)

    norm_x = torch.cat([g.contiguous().view(-1) for g in hvp_x])
    norm_y = torch.cat([g.contiguous().view(-1) for g in hvp_y])

    norm_x = torch.add(grad_x_vec, - lr * norm_x)
    norm_y = torch.add(grad_y_vec, lr * norm_y)

    return torch.norm(grad_x_vec, p=2), torch.norm(grad_y_vec, p=2), \
           torch.norm(norm_x, p=2), torch.norm(norm_y, p=2)


def CGD(loss, max_params, min_params, old_x=None, old_y=None, lr=1e-2, device=torch.device('cpu'), cg_time=False, weight_decay=0, solve_x=False):
    '''
    :param loss: object function
    :param max_params: x
    :param min_params: y
    :param lr: learning rate
    :param weight_decay: weight decay rate recommend value: 0.01, 0.001, 0.0001
    :param solve_x: False - given x solve for y;
                    True  - given y solve for x;
    :return: L2 norm of grad, hvp, equilibrium term

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

    '''
    zero_grad(max_params)
    zero_grad(min_params)

    grad_x = autograd.grad(loss, max_params, create_graph=True, retain_graph=True)
    grad_x_vec = torch.cat([g.contiguous().view(-1) for g in grad_x])
    grad_y = autograd.grad(loss, min_params, create_graph=True, retain_graph=True)
    grad_y_vec = torch.cat([g.contiguous().view(-1) for g in grad_y])

    hvp_x_vec = Hvp_vec(grad_y_vec, max_params, grad_y_vec, retain_graph=True) # h_xy * d_y
    hvp_y_vec = Hvp_vec(grad_x_vec, min_params, grad_x_vec, retain_graph=True) # h_yx * d_x

    p_x = torch.add(grad_x_vec, - lr * hvp_x_vec)
    p_y = torch.add(grad_y_vec, lr * hvp_y_vec)

    norm_x = torch.norm(p_x, p=2)
    norm_y = torch.norm(p_y, p=2)
    re_x = None
    re_y = None
    if cg_time:
        timer_x = time.time()
    if solve_x:
        cg_y, iter_num = conjugate_gradient(grad_x=grad_y_vec, grad_y=grad_x_vec, x_params=min_params, y_params=max_params,
                                          b=p_y, x=old_y, nsteps=p_y.shape[0], lr=lr, device=device)
        hcg = Hvp_vec(grad_y_vec, max_params, cg_y)
        cg_x = torch.add(grad_x_vec, - lr * hcg)
        re_x = cg_x
    else:
        cg_x, iter_num = conjugate_gradient(grad_x=grad_x_vec, grad_y=grad_y_vec, x_params=max_params, y_params=min_params,
                                          b=p_x, x=old_x, nsteps=p_x.shape[0], lr=lr, device=device)
        hcg = Hvp_vec(grad_x_vec, min_params, cg_x)
        cg_y = torch.add(grad_y_vec, lr * hcg)
        re_y = cg_y
    if cg_time:
        timer_x = time.time() - timer_x

    index = 0
    for p in max_params:
        if weight_decay != 0:
            p.data.add_(- weight_decay * p)
        p.data.add_(lr * cg_x[index : index + p.numel()].reshape(p.shape))
        index += p.numel()
    if index != cg_x.numel():
        raise ValueError('CG size mismatch')
    index = 0
    for p in min_params:
        if weight_decay != 0:
            p.data.add_(- weight_decay * p)
        p.data.add_(- lr * cg_y[index : index + p.numel()].reshape(p.shape))
        index += p.numel()
    if index != cg_y.numel():
        raise ValueError('CG size mismatch')
    if cg_time:
        return torch.norm(grad_x_vec, p=2), torch.norm(grad_y_vec, p=2), \
               norm_x, norm_y, \
               torch.norm(cg_x, p=2), torch.norm(cg_y, p=2), \
               timer_x, iter_num, re_x, re_y

    return torch.norm(grad_x_vec, p=2), torch.norm(grad_y_vec, p=2), \
           norm_x, norm_y, \
           torch.norm(cg_x, p=2), torch.norm(cg_y, p=2), re_x, re_y


class BCGD(object):
    def __init__(self, max_params, min_params, lr=1e-3, weight_decay=0, device=torch.device('cpu'),solve_x=False, collect_info=True):
        self.max_params = max_params
        self.min_params = min_params
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.solve_x = solve_x
        self.collect_info = collect_info

        self.old_x = None
        self.old_y = None

    def zero_grad(self):
        zero_grad(self.max_params)
        zero_grad(self.min_params)

    def getinfo(self):
        if self.collect_info:
            return self.norm_gx, self.norm_gy, self.norm_px, self.norm_py, self.norm_cgx, self.norm_cgy, \
                   self.timer, self.iter_num
        else:
            raise ValueError('No update information stored. Set collect_info=True before call this method')

    def step(self, loss):
        grad_x = autograd.grad(loss, self.max_params, create_graph=True, retain_graph=True)
        grad_x_vec = torch.cat([g.contiguous().view(-1) for g in grad_x])
        grad_y = autograd.grad(loss, self.min_params, create_graph=True, retain_graph=True)
        grad_y_vec = torch.cat([g.contiguous().view(-1) for g in grad_y])

        hvp_x_vec = Hvp_vec(grad_y_vec, self.max_params, grad_y_vec, retain_graph=True)  # h_xy * d_y
        hvp_y_vec = Hvp_vec(grad_x_vec, self.min_params, grad_x_vec, retain_graph=True)  # h_yx * d_x

        p_x = torch.add(grad_x_vec, - self.lr * hvp_x_vec)
        p_y = torch.add(grad_y_vec, self.lr * hvp_y_vec)
        if self.collect_info:
            self.norm_px = torch.norm(p_x, p=2)
            self.norm_py = torch.norm(p_y, p=2)
            self.timer = time.time()
        if self.solve_x:
            cg_y, self.iter_num = conjugate_gradient(grad_x=grad_y_vec, grad_y=grad_x_vec, x_params=self.min_params,
                                                y_params=self.max_params, b=p_y, x=self.old_y, nsteps=p_y.shape[0]//10000,
                                                lr=self.lr, device=self.device)
            hcg = Hvp_vec(grad_y_vec, self.max_params, cg_y)
            cg_x = torch.add(grad_x_vec, - self.lr * hcg)
            self.old_x = cg_x
        else:
            cg_x, self.iter_num = conjugate_gradient(grad_x=grad_x_vec, grad_y=grad_y_vec, x_params=self.max_params,
                                                y_params=self.min_params, b=p_x, x=self.old_x, nsteps=p_x.shape[0]//10000,
                                                lr=self.lr, device=self.device)
            hcg = Hvp_vec(grad_x_vec, self.min_params, cg_x)
            cg_y = torch.add(grad_y_vec, self.lr * hcg)
            self.old_y = cg_y

        if self.collect_info:
            self.timer = time.time() - self.timer

        index = 0
        for p in self.max_params:
            if self.weight_decay != 0:
                p.data.add_(- self.weight_decay * p)
            p.data.add_(self.lr * cg_x[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != cg_x.numel():
            raise ValueError('CG size mismatch')
        index = 0
        for p in self.min_params:
            if self.weight_decay != 0:
                p.data.add_(- self.weight_decay * p)
            p.data.add_(- self.lr * cg_y[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != cg_y.numel():
            raise ValueError('CG size mismatch')

        if self.collect_info:
            self.norm_gx = torch.norm(grad_x_vec, p=2)
            self.norm_gy = torch.norm(grad_y_vec, p=2)
            self.norm_cgx = torch.norm(cg_x, p=2)
            self.norm_cgy = torch.norm(cg_y, p=2)
        self.solve_x = False if self.solve_x else True


class ACGD(object):
    def __init__(self, max_params, min_params, eps=1e-5, beta2=0.99, lr=1e-3, weight_decay=0, device=torch.device('cpu'),solve_x=False, collect_info=True):
        self.max_params = max_params
        self.min_params = min_params
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.solve_x = solve_x
        self.collect_info = collect_info
        self.square_avgx = None
        self.square_avgy = None
        self.beta2 = beta2
        self.eps = eps
        self.cg_x = None
        self.cg_y = None

        self.old_x = None
        self.old_y = None

    def zero_grad(self):
        zero_grad(self.max_params)
        zero_grad(self.min_params)

    def getinfo(self):
        if self.collect_info:
            return self.norm_gx, self.norm_gy, self.norm_px, self.norm_py, self.norm_cgx, self.norm_cgy, \
                   self.timer, self.iter_num
        else:
            raise ValueError('No update information stored. Set get_norms True before call this method')

    def get_lrs(self):
        raise NotImplementedError

    def step(self, loss):
        grad_x = autograd.grad(loss, self.max_params, create_graph=True, retain_graph=True)
        grad_x_vec = torch.cat([g.contiguous().view(-1) for g in grad_x])
        grad_y = autograd.grad(loss, self.min_params, create_graph=True, retain_graph=True)
        grad_y_vec = torch.cat([g.contiguous().view(-1) for g in grad_y])

        if self.square_avgx is None and self.square_avgy is None:
            # self.square_avgx = torch.mul(grad_x_vec.data, grad_x_vec.data)
            # self.square_avgy = torch.mul(grad_y_vec.data, grad_y_vec.data)
            self.square_avgx = torch.zeros(grad_x_vec.size(), requires_grad=False, device=self.device)
            # self.square_avgx /= self.square_avgx.norm(p=1)
            self.square_avgy = torch.zeros(grad_y_vec.size(), requires_grad=False, device=self.device)
            # self.square_avgy /= self.square_avgy.norm(p=1)
        # if self.cg_x is None and self.cg_y is None:
        #     self.cg_x = torch.ones_like(grad_x_vec.data)
        #     self.cg_y = torch.ones_like(grad_y_vec.data)
        # self.square_avgx.mul_(self.beta2).addcmul_(1 - self.beta2, self.cg_x, self.cg_x)
        # self.square_avgy.mul_(self.beta2).addcmul_(1 - self.beta2, self.cg_y, self.cg_y)
        self.square_avgx.mul_(self.beta2).addcmul_(1 - self.beta2, grad_x_vec.data, grad_x_vec.data)
        self.square_avgy.mul_(self.beta2).addcmul_(1 - self.beta2, grad_y_vec.data, grad_y_vec.data)
        lr_x = self.lr / self.square_avgx.sqrt().add_(self.eps)
        lr_y = self.lr / self.square_avgy.sqrt().add_(self.eps)

        scaled_grad_x = torch.mul(lr_x, grad_x_vec).detach() # lr_x * grad_x
        scaled_grad_y = torch.mul(lr_y, grad_y_vec).detach() # lr_y * grad_y
        hvp_x_vec = Hvp_vec(grad_y_vec, self.max_params, scaled_grad_y, retain_graph=True)  # D_xy * lr_y * grad_y
        hvp_y_vec = Hvp_vec(grad_x_vec, self.min_params, scaled_grad_x, retain_graph=True)  # D_yx * lr_x * grad_x

        p_x = torch.add(grad_x_vec, - hvp_x_vec) #grad_x - D_xy * lr_y * grad_y
        p_y = torch.add(grad_y_vec, hvp_y_vec)   #grad_y + D_yx * lr_x * grad_x

        if self.collect_info:
            self.norm_px = torch.norm(hvp_x_vec, p=2).detach()
            self.norm_py = torch.norm(hvp_y_vec, p=2).detach()
            self.timer = time.time()
        # p_x.mul_(lr_x)
        # p_y.mul_(lr_y)
        if self.solve_x:
            p_y.mul_(lr_y.sqrt())
            p_y_norm = p_y.norm(p=2).detach()
            if self.old_y is not None:
                self.old_y = self.old_y / p_y_norm
            if torch.isnan(p_y).any():
                path = 'checkpoints/nan2.pth'
                print('save parameters at %s \n b=p_y, x=old_y' % path)
                torch.save({
                    'D': self.min_params,
                    'G': self.max_params,
                    'D gradient': grad_y_vec,
                    'G gradient': grad_x_vec,
                    'x': self.old_y,
                    'b': p_y
                }, path)
                raise ValueError('p_y is nan')
            cg_y, self.iter_num = general_conjugate_gradient(grad_x=grad_y_vec, grad_y=grad_x_vec,
                                                             x_params=self.min_params,
                                                             y_params=self.max_params, b=p_y / p_y_norm, x=self.old_y,
                                                             nsteps=p_y.shape[0] // 10000,
                                                             lr_x=lr_y, lr_y=lr_x, device=self.device)
            cg_y.mul_(p_y_norm)
            # try:
            #     cg_y, self.iter_num = general_conjugate_gradient(grad_x=grad_y_vec, grad_y=grad_x_vec, x_params=self.min_params,
            #                                     y_params=self.max_params, b=p_y, x=self.old_y, nsteps=p_y.shape[0] // 10000,
            #                                     lr_x=lr_y, lr_y=lr_x, device=self.device)
            # (I + lr_y.sqrt() * D_yx * lr_x * D_xy * lr_y.sqrt()) ** -1 * lr_y.sqrt() * p_y
            cg_y.mul_(- lr_y.sqrt())
            hcg = Hvp_vec(grad_y_vec, self.max_params, cg_y, retain_graph=False).add_(grad_x_vec)
            # grad_x + D_xy * delta y
            cg_x = hcg.mul(lr_x)
            self.old_x = hcg.mul(lr_x.sqrt())
        else:
            p_x.mul_(lr_x.sqrt())
            p_x_norm = p_x.norm(p=2).detach()
            if self.old_x is not None:
                self.old_x = self.old_x / p_x_norm
            if torch.isnan(p_x).any():
                path = 'checkpoints/nan2.pth'
                print('save parameters at %s \n b=p_x, x=old_x' % path)
                torch.save({
                    'D': self.min_params,
                    'G': self.max_params,
                    'D gradient': grad_y_vec,
                    'G gradient': grad_x_vec,
                    'x': self.old_x,
                    'b': p_x
                }, path)
                raise ValueError('px is nan')
            cg_x, self.iter_num = general_conjugate_gradient(grad_x=grad_x_vec, grad_y=grad_y_vec,
                                                             x_params=self.max_params,
                                                             y_params=self.min_params, b=p_x / p_x_norm, x=self.old_x,
                                                             nsteps=p_x.shape[0] // 10000,
                                                             lr_x=lr_x, lr_y=lr_y, device=self.device)
            cg_x.mul_(p_x_norm)
            # try:
            #     cg_x, self.iter_num = general_conjugate_gradient(grad_x=grad_x_vec, grad_y=grad_y_vec, x_params=self.max_params,
            #                                     y_params=self.min_params, b=p_x, x=self.old_x, nsteps=p_x.shape[0] // 10000,
            #                                     lr_x=lr_x, lr_y=lr_y, device=self.device)
            # except:
            #     path = 'checkpoints/nan.pth'
            #     print('save parameters at %s \n b=p_x, x=old_x' % path)
            #     torch.save({
            #         'D': self.min_params,
            #         'G': self.max_params,
            #         'D gradient': grad_y_vec,
            #         'G gradient': grad_x_vec,
            #         'x': self.old_x,
            #         'b': p_x
            #     }, path)
            # (I + lr_x.sqrt() * D_xy * lr_y * D_yx * lr_x.sqrt()) ** -1 * lr_x.sqrt() * p_x
            cg_x.mul_(lr_x.sqrt()) # delta x = lr_x.sqrt() * cg_x
            hcg = Hvp_vec(grad_x_vec, self.min_params, cg_x, retain_graph=True).add_(grad_y_vec)
            # grad_y + D_yx * delta x
            cg_y = hcg.mul(- lr_y)
            self.old_y = hcg.mul(lr_y.sqrt())

        if self.collect_info:
            self.timer = time.time() - self.timer

        index = 0
        for p in self.max_params:
            if self.weight_decay != 0:
                p.data.add_(- self.weight_decay * p)
            p.data.add_(cg_x[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != cg_x.numel():
            raise RuntimeError('CG size mismatch')
        index = 0
        for p in self.min_params:
            if self.weight_decay != 0:
                p.data.add_(- self.weight_decay * p)
            p.data.add_(cg_y[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != cg_y.numel():
            raise RuntimeError('CG size mismatch')
        # self.cg_x = cg_x.detach()
        # self.cg_y = cg_y.detach()
        if self.collect_info:
            self.norm_gx = torch.norm(grad_x_vec, p=2)
            self.norm_gy = torch.norm(grad_y_vec, p=2)
            self.norm_cgx = torch.norm(cg_x, p=2)
            self.norm_cgy = torch.norm(cg_y, p=2)

        self.solve_x = False if self.solve_x else True



class OCGD(object):
    def __init__(self, max_params, min_params, eps=1e-5, beta2=0.99, lr=1e-3, device=torch.device('cpu'),
                 update_min=False, collect_info=True):
        self.max_params = max_params
        self.min_params = min_params
        self.lr = lr
        self.device = device
        self.update_min = update_min
        self.collect_info = collect_info
        self.avgx_sq = None
        self.avgy_sq = None
        self.avgx = None
        self.avgy = None
        self.beta2 = beta2
        self.eps = eps

        self.old_x = None
        self.old_y = None

    def zero_grad(self):
        zero_grad(self.max_params)
        zero_grad(self.min_params)

    def getinfo(self):
        if self.collect_info:
            return self.norm_gx, self.norm_gy, self.norm_px, self.norm_py, self.norm_cgx, self.norm_cgy, \
                   self.timer, self.iter_num
        else:
            raise ValueError('No update information stored. Set get_norms True before call this method')

    def step(self, loss):
        grad_x = autograd.grad(loss, self.max_params, create_graph=True, retain_graph=True)
        grad_x_vec = torch.cat([g.contiguous().view(-1) for g in grad_x])
        grad_y = autograd.grad(loss, self.min_params, create_graph=True, retain_graph=True)
        grad_y_vec = torch.cat([g.contiguous().view(-1) for g in grad_y])

        if self.avgx_sq is None and self.avgy_sq is None:
            self.avgx_sq = torch.zeros(grad_x_vec.size(), requires_grad=False, device=self.device)
            self.avgy_sq = torch.zeros(grad_y_vec.size(), requires_grad=False, device=self.device)

        self.avgx_sq.mul_(self.beta2).addcmul_(1 - self.beta2, grad_x_vec.data, grad_x_vec.data)
        self.avgy_sq.mul_(self.beta2).addcmul_(1 - self.beta2, grad_y_vec.data, grad_y_vec.data)
        lr_x = self.lr / self.avgx_sq.sqrt().add_(self.eps)
        lr_y = self.lr / self.avgy_sq.sqrt().add_(self.eps)

        scaled_grad_x = torch.mul(lr_x, grad_x_vec).detach()  # lr_x * grad_x
        scaled_grad_y = torch.mul(lr_y, grad_y_vec).detach()  # lr_y * grad_y
        hvp_x_vec = Hvp_vec(grad_y_vec, self.max_params, scaled_grad_y, retain_graph=True)  # D_xy * lr_y * grad_y
        hvp_y_vec = Hvp_vec(grad_x_vec, self.min_params, scaled_grad_x, retain_graph=True)  # D_yx * lr_x * grad_x

        p_x = torch.add(grad_x_vec, - hvp_x_vec)  # grad_x - D_xy * lr_y * grad_y
        p_y = torch.add(grad_y_vec, hvp_y_vec)  # grad_y + D_yx * lr_x * grad_x

        if self.collect_info:
            self.norm_px = lr_x.mean()
            self.norm_py = lr_y.mean()
            self.timer = time.time()

        if self.update_min:
            p_y.mul_(lr_y.sqrt())
            cg_y, self.iter_num = general_conjugate_gradient(grad_x=grad_y_vec, grad_y=grad_x_vec,
                                                             x_params=self.min_params,
                                                             y_params=self.max_params, b=p_y, x=self.old_y,
                                                             nsteps=p_y.shape[0] // 10000,
                                                             lr_x=lr_y, lr_y=lr_x, device=self.device)
            # (I + lr_y.sqrt() * D_yx * lr_x * D_xy * lr_y.sqrt()) ** -1 * lr_y.sqrt() * p_y
            cg_y.mul_(- lr_y.sqrt())
            index = 0
            for p in self.min_params:
                p.data.add_(cg_y[index: index + p.numel()].reshape(p.shape))
                index += p.numel()
            if index != cg_y.numel():
                raise RuntimeError('CG size mismatch')
        else:
            p_x.mul_(lr_x.sqrt())
            cg_x, self.iter_num = general_conjugate_gradient(grad_x=grad_x_vec, grad_y=grad_y_vec,
                                                             x_params=self.max_params,
                                                             y_params=self.min_params, b=p_x, x=self.old_x,
                                                             nsteps=p_x.shape[0] // 10000,
                                                             lr_x=lr_x, lr_y=lr_y, device=self.device)
            # (I + lr_x.sqrt() * D_xy * lr_y * D_yx * lr_x.sqrt()) ** -1 * lr_x.sqrt() * p_x
            cg_x.mul_(lr_x.sqrt())  # delta x = lr_x.sqrt() * cg_x
            index = 0
            for p in self.max_params:
                p.data.add_(cg_x[index: index + p.numel()].reshape(p.shape))
                index += p.numel()
            if index != cg_x.numel():
                raise RuntimeError('CG size mismatch')

        if self.collect_info:
            self.timer = time.time() - self.timer
            self.norm_gx = torch.norm(grad_x_vec, p=2)
            self.norm_gy = torch.norm(grad_y_vec, p=2)
            self.norm_cgx = 0
            self.norm_cgy = 0


class MCGD(object): # allow multi GPU
    def __init__(self, max_params, min_params, eps=1e-8, beta2=0.99, lr=1e-3, device=torch.device('cpu'),solve_x=False, collect_info=True):
        self.max_params = max_params
        self.min_params = min_params
        self.lr = lr
        self.device = device
        self.solve_x = solve_x
        self.collect_info = collect_info
        self.square_avgx = None
        self.square_avgy = None
        self.beta2 = beta2
        self.eps = eps
        self.cg_x = None
        self.cg_y = None
        self.count = 0

        self.old_x = None
        self.old_y = None

    def zero_grad(self):
        zero_grad(self.max_params.parameters())
        zero_grad(self.min_params.parameters())

    def getinfo(self):
        if self.collect_info:
            return self.norm_gx, self.norm_gy, self.norm_px, self.norm_py, self.norm_cgx, self.norm_cgy, \
                   self.timer, self.iter_num
        else:
            raise ValueError('No update information stored. Set get_norms True before call this method')

    def step(self, loss):
        self.count += 1
        grad_x = autograd.grad(loss, self.max_params.parameters(), create_graph=True, retain_graph=True)
        grad_x_vec = torch.cat([g.contiguous().view(-1) for g in grad_x])
        grad_y = autograd.grad(loss, self.min_params.parameters(), create_graph=True, retain_graph=True)
        grad_y_vec = torch.cat([g.contiguous().view(-1) for g in grad_y])

        if self.square_avgx is None and self.square_avgy is None:
            self.square_avgx = torch.zeros(grad_x_vec.size(), requires_grad=False, device=self.device)
            self.square_avgy = torch.zeros(grad_y_vec.size(), requires_grad=False, device=self.device)
        self.square_avgx.mul_(self.beta2).addcmul_(1 - self.beta2, grad_x_vec.data, grad_x_vec.data)
        self.square_avgy.mul_(self.beta2).addcmul_(1 - self.beta2, grad_y_vec.data, grad_y_vec.data)

        # Initialization bias correction
        bias_correction2 = 1 - self.beta2 ** self.count

        lr_x = math.sqrt(bias_correction2) * self.lr  / self.square_avgx.sqrt().add(self.eps)
        lr_y = math.sqrt(bias_correction2) * self.lr / self.square_avgy.sqrt().add(self.eps)
        scaled_grad_x = torch.mul(lr_x, grad_x_vec).detach() # lr_x * grad_x
        scaled_grad_y = torch.mul(lr_y, grad_y_vec).detach() # lr_y * grad_y
        hvp_x_vec = Hvpvec(grad_y_vec, self.max_params, scaled_grad_y, retain_graph=True)  # D_xy * lr_y * grad_y
        hvp_y_vec = Hvpvec(grad_x_vec, self.min_params, scaled_grad_x, retain_graph=True)  # D_yx * lr_x * grad_x

        p_x = torch.add(grad_x_vec, - hvp_x_vec).detach_() #grad_x - D_xy * lr_y * grad_y
        p_y = torch.add(grad_y_vec, hvp_y_vec).detach_()   #grad_y + D_yx * lr_x * grad_x

        if self.collect_info:
            self.norm_px = lr_x.max()
            self.norm_py = lr_y.max()
            self.timer = time.time()
        if self.solve_x:
            p_y.mul_(lr_y.sqrt())
            # p_y_norm = p_y.norm(p=2).detach_()
            # if self.old_y is not None:
            #     self.old_y = self.old_y / p_y_norm
            cg_y, self.iter_num = mgeneral_conjugate_gradient(grad_x=grad_y_vec, grad_y=grad_x_vec,
                                                             x_params=self.min_params,
                                                             y_params=self.max_params, b=p_y, x=self.old_y,
                                                             nsteps=p_y.shape[0] // 10000,
                                                             lr_x=lr_y, lr_y=lr_x, device=self.device)
            # cg_y.mul_(p_y_norm)
            cg_y.detach_().mul_(- lr_y.sqrt())
            hcg = Hvpvec(grad_y_vec, self.max_params, cg_y, retain_graph=True).add_(grad_x_vec).detach_()
            # grad_x + D_xy * delta y
            cg_x = hcg.mul(lr_x)
            self.old_x = hcg.mul(lr_x.sqrt())
        else:
            p_x.mul_(lr_x.sqrt())
            # p_x_norm = p_x.norm(p=2).detach_()
            # if self.old_x is not None:
            #     self.old_x = self.old_x / p_x_norm
            cg_x, self.iter_num = mgeneral_conjugate_gradient(grad_x=grad_x_vec, grad_y=grad_y_vec,
                                                             x_params=self.max_params,
                                                             y_params=self.min_params, b=p_x, x=self.old_x,
                                                             nsteps=p_x.shape[0] // 10000,
                                                             lr_x=lr_x, lr_y=lr_y, device=self.device)
            # cg_x.detach_().mul_(p_x_norm)
            cg_x.detach_().mul_(lr_x.sqrt()) # delta x = lr_x.sqrt() * cg_x
            hcg = Hvpvec(grad_x_vec, self.min_params, cg_x, retain_graph=True).add_(grad_y_vec).detach_()
            # grad_y + D_yx * delta x
            cg_y = hcg.mul(- lr_y)
            self.old_y = hcg.mul(lr_y.sqrt())

        if self.collect_info:
            self.timer = time.time() - self.timer

        index = 0
        for p in self.max_params.parameters():
            p.data.add_(cg_x[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != cg_x.numel():
            raise RuntimeError('CG size mismatch')
        index = 0
        for p in self.min_params.parameters():
            p.data.add_(cg_y[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != cg_y.numel():
            raise RuntimeError('CG size mismatch')
        if self.collect_info:
            self.norm_gx = torch.norm(grad_x_vec, p=2)
            self.norm_gy = torch.norm(grad_y_vec, p=2)
            self.norm_cgx = torch.norm(cg_x, p=2)
            self.norm_cgy = torch.norm(cg_y, p=2)

        self.solve_x = False if self.solve_x else True
