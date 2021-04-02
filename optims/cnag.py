import time
import torch
from torch.optim.optimizer import Optimizer


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










