import math
import time

import torch
import torch.autograd as autograd

from cgd_utils import conjugate_gradient, Hvp_vec, general_conjugate_gradient, Hvpvec, \
    mgeneral_conjugate_gradient, zero_grad


class BCGD(object):
    def __init__(self, max_params, min_params, lr=1e-3, weight_decay=0, device=torch.device('cpu'),
                 solve_x=False, collect_info=True):
        self.max_params = list(max_params)
        self.min_params = list(min_params)
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
            raise ValueError(
                'No update information stored. Set collect_info=True before call this method')

    def step(self, loss):
        grad_x = autograd.grad(loss, self.max_params, create_graph=True, retain_graph=True)
        grad_x_vec = torch.cat([g.contiguous().view(-1) for g in grad_x])
        grad_y = autograd.grad(loss, self.min_params, create_graph=True, retain_graph=True)
        grad_y_vec = torch.cat([g.contiguous().view(-1) for g in grad_y])

        hvp_x_vec = Hvp_vec(grad_y_vec, self.max_params, grad_y_vec,
                            retain_graph=True)  # h_xy * d_y
        hvp_y_vec = Hvp_vec(grad_x_vec, self.min_params, grad_x_vec,
                            retain_graph=True)  # h_yx * d_x

        p_x = torch.add(grad_x_vec, - self.lr * hvp_x_vec)
        p_y = torch.add(grad_y_vec, self.lr * hvp_y_vec)
        if self.collect_info:
            self.norm_px = torch.norm(p_x, p=2)
            self.norm_py = torch.norm(p_y, p=2)
            self.timer = time.time()
        if self.solve_x:
            cg_y, self.iter_num = conjugate_gradient(grad_x=grad_y_vec, grad_y=grad_x_vec,
                                                     x_params=self.min_params,
                                                     y_params=self.max_params, b=p_y, x=self.old_y,
                                                     nsteps=p_y.shape[0] // 10000,
                                                     lr=self.lr, device=self.device)
            hcg = Hvp_vec(grad_y_vec, self.max_params, cg_y)
            cg_x = torch.add(grad_x_vec, - self.lr * hcg)
            self.old_x = cg_x
        else:
            cg_x, self.iter_num = conjugate_gradient(grad_x=grad_x_vec, grad_y=grad_y_vec,
                                                     x_params=self.max_params,
                                                     y_params=self.min_params, b=p_x, x=self.old_x,
                                                     nsteps=p_x.shape[0] // 10000,
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


class OCGD(object):
    def __init__(self, max_params, min_params, eps=1e-5, beta2=0.99, lr=1e-3,
                 device=torch.device('cpu'),
                 update_min=False, collect_info=True):
        self.max_params = list(max_params)
        self.min_params = list(min_params)
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
            raise ValueError(
                'No update information stored. Set get_norms True before call this method')

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
        hvp_x_vec = Hvp_vec(grad_y_vec, self.max_params, scaled_grad_y,
                            retain_graph=True)  # D_xy * lr_y * grad_y
        hvp_y_vec = Hvp_vec(grad_x_vec, self.min_params, scaled_grad_x,
                            retain_graph=True)  # D_yx * lr_x * grad_x

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
                                                             y_params=self.max_params, b=p_y,
                                                             x=self.old_y,
                                                             nsteps=p_y.shape[0] // 10000,
                                                             lr_x=lr_y, lr_y=lr_x,
                                                             device=self.device)
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
                                                             y_params=self.min_params, b=p_x,
                                                             x=self.old_x,
                                                             nsteps=p_x.shape[0] // 10000,
                                                             lr_x=lr_x, lr_y=lr_y,
                                                             device=self.device)
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


class ACGD(object):  # Support multi GPU
    def __init__(self, max_params, min_params, eps=1e-8, beta2=0.99, lr=1e-3,
                 device=torch.device('cpu'), solve_x=False, collect_info=True):
        self.max_params = list(max_params)
        self.min_params = list(min_params)
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
        zero_grad(self.max_params)
        zero_grad(self.min_params)

    def getinfo(self):
        if self.collect_info:
            return self.norm_gx, self.norm_gy, self.norm_px, self.norm_py, self.norm_cgx, self.norm_cgy, \
                   self.timer, self.iter_num
        else:
            raise ValueError(
                'No update information stored. Set get_norms True before call this method')

    def step(self, loss):
        self.count += 1
        grad_x = autograd.grad(loss, self.max_params, create_graph=True,
                               retain_graph=True)
        grad_x_vec = torch.cat([g.contiguous().view(-1) for g in grad_x])
        grad_y = autograd.grad(loss, self.min_params, create_graph=True,
                               retain_graph=True)
        grad_y_vec = torch.cat([g.contiguous().view(-1) for g in grad_y])

        if self.square_avgx is None and self.square_avgy is None:
            self.square_avgx = torch.zeros(grad_x_vec.size(), requires_grad=False,
                                           device=self.device)
            self.square_avgy = torch.zeros(grad_y_vec.size(), requires_grad=False,
                                           device=self.device)
        self.square_avgx.mul_(self.beta2).addcmul_(1 - self.beta2, grad_x_vec.data, grad_x_vec.data)
        self.square_avgy.mul_(self.beta2).addcmul_(1 - self.beta2, grad_y_vec.data, grad_y_vec.data)

        # Initialization bias correction
        bias_correction2 = 1 - self.beta2 ** self.count

        lr_x = math.sqrt(bias_correction2) * self.lr / self.square_avgx.sqrt().add(self.eps)
        lr_y = math.sqrt(bias_correction2) * self.lr / self.square_avgy.sqrt().add(self.eps)
        scaled_grad_x = torch.mul(lr_x, grad_x_vec).detach()  # lr_x * grad_x
        scaled_grad_y = torch.mul(lr_y, grad_y_vec).detach()  # lr_y * grad_y
        hvp_x_vec = Hvp_vec(grad_y_vec, self.max_params, scaled_grad_y,
                           retain_graph=True)  # D_xy * lr_y * grad_y
        hvp_y_vec = Hvp_vec(grad_x_vec, self.min_params, scaled_grad_x,
                           retain_graph=True)  # D_yx * lr_x * grad_x

        p_x = torch.add(grad_x_vec, - hvp_x_vec).detach_()  # grad_x - D_xy * lr_y * grad_y
        p_y = torch.add(grad_y_vec, hvp_y_vec).detach_()  # grad_y + D_yx * lr_x * grad_x

        if self.collect_info:
            self.norm_px = lr_x.max()
            self.norm_py = lr_y.max()
            self.timer = time.time()
        if self.solve_x:
            p_y.mul_(lr_y.sqrt())
            # p_y_norm = p_y.norm(p=2).detach_()
            # if self.old_y is not None:
            #     self.old_y = self.old_y / p_y_norm
            cg_y, self.iter_num = general_conjugate_gradient(grad_x=grad_y_vec, grad_y=grad_x_vec,
                                                             x_params=self.min_params,
                                                             y_params=self.max_params, b=p_y,
                                                             x=self.old_y,
                                                             nsteps=p_y.shape[0] // 10000,
                                                             lr_x=lr_y, lr_y=lr_x,
                                                             device=self.device)
            # cg_y.mul_(p_y_norm)
            cg_y.detach_().mul_(- lr_y.sqrt())
            hcg = Hvp_vec(grad_y_vec, self.max_params, cg_y, retain_graph=True).add_(
                grad_x_vec).detach_()
            # grad_x + D_xy * delta y
            cg_x = hcg.mul(lr_x)
            self.old_x = hcg.mul(lr_x.sqrt())
        else:
            p_x.mul_(lr_x.sqrt())
            # p_x_norm = p_x.norm(p=2).detach_()
            # if self.old_x is not None:
            #     self.old_x = self.old_x / p_x_norm
            cg_x, self.iter_num = general_conjugate_gradient(grad_x=grad_x_vec, grad_y=grad_y_vec,
                                                             x_params=self.max_params,
                                                             y_params=self.min_params, b=p_x,
                                                             x=self.old_x,
                                                             nsteps=p_x.shape[0] // 10000,
                                                             lr_x=lr_x, lr_y=lr_y,
                                                             device=self.device)
            # cg_x.detach_().mul_(p_x_norm)
            cg_x.detach_().mul_(lr_x.sqrt())  # delta x = lr_x.sqrt() * cg_x
            hcg = Hvp_vec(grad_x_vec, self.min_params, cg_x, retain_graph=True).add_(
                grad_y_vec).detach_()
            # grad_y + D_yx * delta x
            cg_y = hcg.mul(- lr_y)
            self.old_y = hcg.mul(lr_y.sqrt())

        if self.collect_info:
            self.timer = time.time() - self.timer

        index = 0
        for p in self.max_params:
            p.data.add_(cg_x[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != cg_x.numel():
            raise RuntimeError('CG size mismatch')
        index = 0
        for p in self.min_params:
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
