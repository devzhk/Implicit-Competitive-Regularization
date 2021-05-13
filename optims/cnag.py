import math
import torch
from torch.optim.optimizer import Optimizer


class CNAG(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(CNAG, self).__init__(params, defaults)
        self.init_state()

    def __setstate__(self, state):
        super(CNAG, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def init_state(self):
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'last_delta' not in param_state:
                    param_state['step'] = 0
                    param_state['last_delta'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    # Exponential moving average of gradient values
                    param_state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    param_state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)


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
                grad = p.grad.data

                param_state = self.state[p]

                exp_avg, exp_avg_sq = param_state['exp_avg'], param_state['exp_avg_sq']
                beta1, beta2 = group['betas']

                param_state['step'] += 1
                bias_correction1 = 1 - beta1 ** param_state['step']
                bias_correction2 = 1 - beta2 ** param_state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1
                d_p = - step_size * exp_avg / denom
                param_state['last_delta'] = d_p
                # p.data.add_(d_p)
        return loss










