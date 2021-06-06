import time
import math
import torch

from .cgd_utils import zero_grad, conjugate_gradient_precondition, Hvp_vec, compute_grad, Matvec


class SACGD(object):
    def __init__(self, x_params, y_params,
                 lr_x=1e-3, lr_y=1e-3,
                 eps=1e-5, beta=0.99,
                 tol=1e-12, atol=1e-20,
                 device=torch.device('cpu'),
                 solve_x=False, collect_info=True,
                 solver='cg'):
        '''

        :param x_params: generator
        :param y_params: discriminator
        :param lr_x:
        :param lr_y:
        :param eps:
        :param beta:
        :param tol:
        :param atol:
        :param device:
        :param solve_x:
        :param collect_info:
        :param solver:
        '''
        self.x_params = list(x_params)
        self.y_params = list(y_params)
        self.state = {'lr_x': lr_x, 'lr_y': lr_y,
                      'eps': eps, 'solve_x': solve_x,
                      'tol': tol, 'atol': atol,
                      'beta': beta, 'step': 0,
                      'old_x': None, 'old_y': None,  # start point of CG
                      'sq_exp_avg_max': None, 'sq_exp_avg_min': None}  # save last update
        self.info = {'grad_x': None, 'grad_y': None,
                     'hvp_x': None, 'hvp_y': None,
                     'cg_x': None, 'cg_y': None,
                     'time': 0, 'iter_num': 0}
        self.device = device
        self.collect_info = collect_info
        self.solver= solver

    def zero_grad(self):
        zero_grad(self.x_params)
        zero_grad(self.y_params)

    def get_info(self):
        if self.info['grad_x'] is None:
            print('Warning! No update information stored. Set collect_info=True before call this method')
        return self.info

    def state_dict(self):
        return self.state

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)
        print('Load state: {}'.format(state_dict))
        if self.state['sq_exp_avg_max'] is not None:
            self.state['sq_exp_avg_max'] = self.state['sq_exp_avg_max'].to(self.device)
            self.state['sq_exp_avg_min'] = self.state['sq_exp_avg_min'].to(self.device)
            self.state['old_x'] = self.state['old_x'].to(self.device)
            self.state['old_y'] = self.state['old_y'].to(self.device)

    def set_lr(self, lr_x, lr_y):
        ratio_max = math.sqrt(self.state['lr_x'] / lr_x)
        ratio_min = math.sqrt(self.state['lr_y'] / lr_y)
        if abs(ratio_max - 1) > 1e-3 and self.state['old_x'] is not None:
            self.state['old_x'] = self.state['old_x'] * ratio_max
        if abs(ratio_min - 1) > 1e-3 and self.state['old_y'] is not None:
            self.state['old_y'] = self.state['old_y'] * ratio_min
        self.state.update({'lr_x': lr_x, 'lr_y': lr_y})
        # print('Maximizing side learning rate: {:.4f}\n '
        #       'Minimizing side learning rate: {:.4f}'.format(lr_x, lr_y))

    def step(self, lossG, lossD):
        '''
        :param lossG: loss f
        :param lossD: loss g
        '''
        lr_x = self.state['lr_x']
        lr_y = self.state['lr_y']
        beta = self.state['beta']
        eps = self.state['eps']
        tol = self.state['tol']
        atol = self.state['atol']
        time_step = self.state['step'] + 1
        self.state['step'] = time_step

        # get \nabla_x f
        grad_xf_vec = compute_grad(lossG, self.x_params)
        # get \nabla_y g
        grad_yg_vec = compute_grad(lossD, self.y_params)
        # flatten gradients
        grad_xf_vec_d = grad_xf_vec.clone().detach()
        grad_yg_vec_d = grad_yg_vec.clone().detach()

        sq_avg_x = self.state['sq_exp_avg_max']
        sq_avg_y = self.state['sq_exp_avg_min']
        sq_avg_x = torch.zeros_like(grad_xf_vec_d, requires_grad=False) if sq_avg_x is None else sq_avg_x
        sq_avg_y = torch.zeros_like(grad_yg_vec_d, requires_grad=False) if sq_avg_y is None else sq_avg_y

        sq_avg_x.mul_(beta).addcmul_(grad_xf_vec_d, grad_xf_vec_d, value=1 - beta)
        sq_avg_y.mul_(beta).addcmul_(grad_yg_vec_d, grad_yg_vec_d, value=1 - beta)

        bias_correction = 1 - beta ** time_step
        lr_x = math.sqrt(bias_correction) * lr_x / sq_avg_x.sqrt().add(eps)
        lr_y = math.sqrt(bias_correction) * lr_y / sq_avg_y.sqrt().add(eps)
        # \eta_x \nabla_x f
        scaled_grad_xf = torch.mul(lr_x, grad_xf_vec_d)
        # \eta_y \nabla_y g
        scaled_grad_yg = torch.mul(lr_y, grad_yg_vec_d)
        # \nabla_y f
        grad_yf_vec = compute_grad(lossG, self.y_params)
        # \nabla_x g
        grad_xg_vec = compute_grad(lossD, self.x_params)
        
        # D_{xy}^2f \eta_y \nabla_y g
        hvp_x_vec = Hvp_vec(grad_yf_vec, self.x_params, scaled_grad_yg,
                            retain_graph=True)
        # D_{yx}^2g \eta_x \nabla_x f
        hvp_y_vec = Hvp_vec(grad_xg_vec, self.y_params, scaled_grad_xf,
                            retain_graph=True)

        p_x = torch.add(- grad_xf_vec_d, hvp_x_vec).mul_(lr_x)
        p_y = torch.add(- grad_yg_vec_d, hvp_y_vec).mul_(lr_y)

        if self.collect_info:
            # norm_px = torch.norm(hvp_x_vec, p=2).item()
            # norm_py = torch.norm(hvp_y_vec, p=2).item()
            timer = time.time()

        if self.state['solve_x']:
            Avp_y = Matvec(grad_yf=grad_xf_vec, grad_xg=grad_yg_vec,
                           x_params=self.x_params, y_params=self.y_params,
                           vec=p_y, lr_x=lr_y, lr_y=lr_x,
                           transpose=True)
            cg_y, iter_num = conjugate_gradient_precondition(grad_xf=grad_yg_vec, grad_xg=grad_yf_vec,
                                                             grad_yf=grad_xg_vec, grad_yg=grad_xf_vec,
                                                             x_params=self.y_params, y_params=self.x_params,
                                                             b=Avp_y, x0=self.state['old_y'],
                                                             tol=tol, atol=atol,
                                                             lr_x=lr_y, lr_y=lr_x,
                                                             nsteps=500,
                                                             device=self.device)
            y_update = cg_y.detach_()
            hcg = Hvp_vec(grad_yf_vec, self.x_params, y_update).detach_()
            x_update = - lr_x * (hcg + grad_xf_vec_d)
        else:
            Avp_x = Matvec(grad_yf=grad_yg_vec, grad_xg=grad_xf_vec,
                           x_params=self.x_params, y_params=self.y_params,
                           vec=p_x, lr_x=lr_x, lr_y=lr_y,
                           transpose=True)
            cg_x, iter_num = conjugate_gradient_precondition(grad_xf=grad_xf_vec, grad_xg=grad_xg_vec,
                                                             grad_yf=grad_yf_vec, grad_yg=grad_yg_vec,
                                                             x_params=self.x_params, y_params=self.y_params,
                                                             b=Avp_x, x0=self.state['old_x'],
                                                             tol=tol, atol=atol,
                                                             lr_x=lr_x, lr_y=lr_y,
                                                             nsteps=500,
                                                             device=self.device)
            x_update = cg_x.detach_()
            hcg = Hvp_vec(grad_xg_vec, self.y_params, x_update).detach_()
            y_update = - lr_y * (hcg + grad_yg_vec_d)
        self.state.update({'old_x': x_update, 'old_y': y_update,
                           'sq_exp_avg_max': sq_avg_x, 'sq_exp_avg_min': sq_avg_y})
        if self.collect_info:
            timer = time.time() - timer
            self.info.update({'time': timer, 'iter_num': iter_num})
        index = 0
        for p in self.x_params:
            p.data.add_(x_update[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        assert index == x_update.numel(), 'Maximizer CG size mismatch'

        index = 0
        for p in self.y_params:
            p.data.add_(y_update[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        assert index == y_update.numel(), 'Minimizer CG size mismatch'

        if self.collect_info:
            norm_gx = torch.norm(grad_xf_vec_d, p=2).item()
            norm_gy = torch.norm(grad_yg_vec_d, p=2).item()
            norm_cgx = torch.norm(x_update, p=2).item()
            norm_cgy = torch.norm(y_update, p=2).item()
            self.info.update({'grad_x': norm_gx, 'grad_y': norm_gy,
                              'cg_x': norm_cgx, 'cg_y': norm_cgy})
        self.state['solve_x'] = False if self.state['solve_x'] else True




