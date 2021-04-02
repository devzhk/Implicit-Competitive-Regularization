import time
import torch
import torch.autograd as autograd
from torch.optim.optimizer import Optimizer

# from .cgd_utils import update_params, Hvp_vec, zero_grad


# class CNAG(object):
#     def __init__(self, max_params, min_params,
#                  lr_max=1e-3, lr_min=1e-3,
#                  device=torch.device('cpu'),
#                  collect_info=True):
#         self.max_params = list(max_params)
#         self.min_params = list(min_params)
#         self.state = {'lr_max': lr_max, 'lr_min': lr_min,
#                       'step': 0, 'delta_max': 0.0, 'delta_min': 0.0}  # save last update
#         self.info = {'grad_x': None, 'grad_y': None,
#                      'hvp_x': None, 'hvp_y': None,
#                      'cg_x': None, 'cg_y': None,
#                      'time': 0, 'iter_num': 0}
#         self.device = device
#         self.collect_info = collect_info
#
#     def zero_grad(self):
#         zero_grad(self.max_params)
#         zero_grad(self.min_params)
#
#     def state_dict(self):
#         return self.state
#
#     def load_state_dict(self, state_dict):
#         self.state.update(state_dict)
#
#     def set_state(self, new_state):
#         self.state.update(new_state)
#         print('Current state: {}'.format(new_state))
#
#     def get_info(self):
#         if self.info['grad_x'] is None:
#             print('Warning! No update information stored. Set collect_info=True before call this method')
#         return self.info
#
#     def step(self, loss):
#         lr_max = self.state['lr_max']
#         lr_min = self.state['lr_min']
#
#         time_step = self.state['step'] + 1
#         self.state['step'] = time_step
#
#         grad_x = autograd.grad(loss, self.max_params, create_graph=True, retain_graph=True)
#         grad_x_vec = torch.cat([g.contiguous().view(-1) for g in grad_x])
#         grad_y = autograd.grad(loss, self.min_params, create_graph=True, retain_graph=True)
#         grad_y_vec = torch.cat([g.contiguous().view(-1) for g in grad_y])
#         grad_x_vec_d = grad_x_vec.clone().detach()
#         grad_y_vec_d = grad_y_vec.clone().detach()


class CNAG(Optimizer):
    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super(CNAG, self).__init__(params, defaults)
        self.init_state()

    def __setstate__(self, state):
        super(CNAG, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def init_state(self):
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'last_delta' not in param_state:
                    param_state['last_delta'] = torch.zeros_like(p.data).detach()

    def update_param(self, scalar=1.0):
        '''
        Update model parameters with last delta
        :param scalar: -1.0 step back
        '''
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                delta_p = param_state['last_delta']
                p.data.add_(delta_p, alpha=scalar)

    def step(self, closure=None):
        """Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                param_state = self.state[p]
                param_state['last_delta'] = - group['lr'] * d_p
                p.data.add_(d_p, alpha=-group['lr'])

        return loss










