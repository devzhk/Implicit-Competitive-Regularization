import torch
from .cgd_utils import zero_grad


class testBCGD(object):
    def __init__(self, max_params, min_params,
                 lr_max=1e-3, lr_min=1e-3,
                 device=torch.device('cpu')):
        self.max_params = list(max_params)
        self.min_params = list(min_params)
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.device = device

    def zero_grad(self):
        zero_grad(self.max_params)
        zero_grad(self.min_params)

    def step(self, loss):
        g_param = torch.cat([p.contiguous().view(-1) for p in self.max_params])
        d_param = torch.cat([p.contiguous().view(-1) for p in self.min_params])
        grad_g = torch.tensor([2 * d_param[0].data, 2 * d_param[1].data,
                               d_param[0].data, d_param[1].data])
        grad_d = torch.tensor([2 * g_param[0].data - 3.0 + g_param[2].data,
                               2 * g_param[1].data - 4.0 + g_param[3].data, 0.0])

        hvp_g = torch.tensor([4 * g_param[0].data - 6.0 + 2 * g_param[2].data,
                              4 * g_param[1].data - 8.0 + 2 * g_param[3].data,
                              2 * g_param[0].data - 3.0 + g_param[2].data,
                              2 * g_param[1].data - 4.0 + g_param[3].data])
        hvp_d = torch.tensor([5 * d_param[0].data, 5 * d_param[1].data, 0.0])
        p_g = torch.add(grad_g, - self.lr_min * hvp_g)
        p_d = torch.add(grad_d, self.lr_max * hvp_d)
        factor = self.lr_max * self.lr_min
        A_g = torch.tensor([[1.0 + 4.0 * factor, 0.0, 2.0 * factor, 0],
                            [0.0, 1.0 + 4.0 * factor, 0.0, 2.0 * factor],
                            [2.0 * factor, 0.0, 1.0 + factor, 0.0],
                            [0.0, 2.0 * factor, 0.0, 1.0 + factor]])
        A_d = torch.tensor([[1.0 + 5.0 * factor, 0.0, 0.0],
                            [0.0, 1.0 + 5.0 * factor, 0.0],
                            [0.0, 0.0, 1.0]])
        cg_g, iter_g = inverse(A=A_g, b=p_g, nsteps=p_g.shape[0])
        cg_d, iter_d = inverse(A=A_d, b=p_d, nsteps=p_d.shape[0])
        # cg_g, iter_g = inverse_gt(A=A_g, b=p_g)
        # cg_d, iter_d = inverse_gt(A=A_d, b=p_d)
        index = 0
        for p in self.max_params:
            p.data.add_(self.lr_max * cg_g[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        assert index == cg_g.numel(), 'Maximizer CG size mismatch'
        index =0
        for p in self.min_params:
            p.data.add_(- self.lr_min * cg_d[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        assert index == cg_d.numel(), 'Minimizer CG size mismatch'


class testLCGD(object):
    def __init__(self, max_params, min_params,
                 lr_max=1e-3, lr_min=1e-3,
                 device=torch.device('cpu')):
        self.max_params = list(max_params)
        self.min_params = list(min_params)
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.device = device

    def zero_grad(self):
        zero_grad(self.max_params)
        zero_grad(self.min_params)

    def step(self, loss):
        g_param = torch.cat([p.contiguous().view(-1) for p in self.max_params])
        d_param = torch.cat([p.contiguous().view(-1) for p in self.min_params])
        grad_g = torch.tensor([2 * d_param[0].data, 2 * d_param[1].data,
                               d_param[0].data, d_param[1].data])
        grad_d = torch.tensor([2 * g_param[0].data - 3.0 + g_param[2].data,
                               2 * g_param[1].data - 4.0 + g_param[3].data, 0.0])

        hvp_g = torch.tensor([4 * g_param[0].data - 6.0 + 2 * g_param[2].data,
                              4 * g_param[1].data - 8.0 + 2 * g_param[3].data,
                              2 * g_param[0].data - 3.0 + g_param[2].data,
                              2 * g_param[1].data - 4.0 + g_param[3].data])
        hvp_d = torch.tensor([5 * d_param[0].data, 5 * d_param[1].data, 0.0])
        cg_g = torch.add(grad_g, - self.lr_min * hvp_g)
        cg_d = torch.add(grad_d, self.lr_max * hvp_d)
        index = 0
        for p in self.max_params:
            p.data.add_(self.lr_max * cg_g[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        assert index == cg_g.numel(), 'Maximizer CG size mismatch'
        index =0
        for p in self.min_params:
            p.data.add_(- self.lr_min * cg_d[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        assert index == cg_d.numel(), 'Minimizer CG size mismatch'


def inverse(A, b, x=None, nsteps=10,
            tol=1e-12, atol=1e-20):
    """
    return A ** -1 * b
    : param tol: relative tolerance
    : param atol: absolute tolerance
    """
    if x is None:
        x = torch.zeros_like(b)
    r = b.clone().detach()
    p = r.clone().detach()
    rdotr = torch.dot(r, r)
    residual_tol = tol * rdotr
    pArray = []
    pArray.append(p)
    for i in range(nsteps):
        Avp_ = torch.matmul(A, p)
        pArray.append(Avp_)
        alpha = rdotr / torch.dot(p, Avp_)
        x.data.add_(alpha * p)
        r.data.add_(- alpha * Avp_)
        new_rdotr = torch.dot(r, r)
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
        if rdotr < residual_tol or rdotr < atol:
            break
    return x, i + 1, pArray


def inverse_gt(A, b):
    return torch.matmul(torch.inverse(A), b), 0

