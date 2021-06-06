import torch
import torch.autograd as autograd
import warnings


def conjugate_gradient(grad_x, grad_y,
                       x_params, y_params,
                       b, x=None, nsteps=10,
                       tol=1e-10, atol=1e-16,
                       lr_x=1.0, lr_y=1.0,
                       device=torch.device('cpu')):
    """
    :param grad_x:
    :param grad_y:
    :param x_params:
    :param y_params:
    :param b: vec
    :param nsteps: max number of steps
    :param residual_tol:
    :return: A ** -1 * b

    h_1 = D_yx * p
    h_2 = D_xy * D_yx * p
    A = I + lr_x * D_xy * lr_y * D_yx
    """
    if x is None:
        x = torch.zeros(b.shape[0], device=device)
        r = b.clone().detach()
    else:
        h1 = Hvp_vec(grad_vec=grad_x, params=y_params, vec=x, retain_graph=True).detach_().mul(lr_y)
        h2 = Hvp_vec(grad_vec=grad_y, params=x_params, vec=h1, retain_graph=True).detach_().mul(lr_x)
        Avx = x + h2
        r = b.clone().detach() - Avx

    p = r.clone().detach()
    rdotr = torch.dot(r, r)
    residual_tol = tol * torch.dot(b, b)
    if rdotr < residual_tol or rdotr < atol:
        return x, 1

    for i in range(nsteps):
        # To compute Avp
        h_1 = Hvp_vec(grad_vec=grad_x, params=y_params, vec=p, retain_graph=True).detach_().mul(lr_y)
        h_2 = Hvp_vec(grad_vec=grad_y, params=x_params, vec=h_1, retain_graph=True).detach_().mul(lr_x)
        Avp_ = p + h_2

        alpha = rdotr / torch.dot(p, Avp_)
        x.data.add_(alpha * p)

        r.data.add_(- alpha * Avp_)
        new_rdotr = torch.dot(r, r)
        if new_rdotr < residual_tol or new_rdotr < atol:
            break
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
    if i > 99:
        warnings.warn('CG iter num: %d' % (i + 1))
    return x, i + 1


def Hvp_vec(grad_vec, params, vec, retain_graph=False):
    '''
    Parameters:
        - grad_vec: Tensor of which the Hessian vector product will be computed
        - params: list of params, w.r.t which the Hessian will be computed
        - vec: The "vector" in Hessian vector product
    return: Hessian vector product
    '''
    if torch.isnan(grad_vec).any():
        raise ValueError('Gradvec nan')
    if torch.isnan(vec).any():
        raise ValueError('vector nan')
        # zero padding for None
    grad_grad = autograd.grad(grad_vec, params, grad_outputs=vec, retain_graph=retain_graph,
                              allow_unused=True)
    grad_list = []
    for i, p in enumerate(params):
        if grad_grad[i] is None:
            grad_list.append(torch.zeros_like(p).view(-1))
        else:
            grad_list.append(grad_grad[i].contiguous().view(-1))
    hvp = torch.cat(grad_list)
    if torch.isnan(hvp).any():
        raise ValueError('hvp Nan')
    return hvp


def Matvec(grad_yf, grad_xg,
           x_params, y_params,
           vec, lr_x, lr_y,
           transpose=True):
    '''
    Compute matrix vector production given by:
        (I - \eta_x D_{xy}f \eta_y D_{yx}g) @ vec
    or its transpose:
        (I - D_{xy}f \eta_y D_{yx}g \eta_x) @ vec
    :param grad_yf: \nabla_y f, vector
    :param grad_xg: \nabla_x g, vector
    :param x_params: list of tensors, x parameters
    :param y_params: list of tensors, x parameters
    :param vec: vector of the same length as grad_xg
    :param lr_x: step sizes vector, same length as grad_xg
    :param lr_y: step sizes vector, same length as grad_yf
    :param transpose: True or False, the order of etas is reversed
    :return: vector of the same dimension as vec
    '''
    if transpose:
        h1 = Hvp_vec(grad_vec=grad_xg, params=y_params,
                     vec=lr_x * vec, retain_graph=True)
        h2 = Hvp_vec(grad_vec=grad_yf, params=x_params,
                     vec=lr_y * h1, retain_graph=True)
    else:
        h1 = Hvp_vec(grad_vec=grad_xg, params=y_params,
                     vec=vec, retain_graph=True).mul_(lr_y)
        h2 = Hvp_vec(grad_vec=grad_yf, params=x_params,
                     vec=h1, retain_graph=True).mul_(lr_x)
    return vec - h2


def general_conjugate_gradient(grad_x, grad_y,
                               x_params, y_params, b,
                               lr_x, lr_y, x=None, nsteps=None,
                               tol=1e-10, atol=1e-16,
                               device=torch.device('cpu')):
    '''

    :param grad_x:
    :param grad_y:
    :param x_params:
    :param y_params:
    :param b:
    :param lr_x:
    :param lr_y:
    :param x:
    :param nsteps:
    :param residual_tol:
    :param device:
    :return: (I + sqrt(lr_x) * D_xy * lr_y * D_yx * sqrt(lr_x)) ** -1 * b

    '''
    lr_x = lr_x.sqrt()
    if x is None:
        x = torch.zeros(b.shape[0], device=device)
        r = b.clone()
    else:
        h1 = Hvp_vec(grad_vec=grad_x, params=y_params,
                     vec=lr_x * x, retain_graph=True).mul_(lr_y)
        h2 = Hvp_vec(grad_vec=grad_y, params=x_params,
                     vec=h1, retain_graph=True).mul_(lr_x)
        Avx = x + h2
        r = b.clone() - Avx

    if nsteps is None:
        nsteps = b.shape[0]

    if grad_x.shape != b.shape:
        raise RuntimeError('CG: hessian vector product shape mismatch')

    p = r.clone().detach()
    rdotr = torch.dot(r, r)
    residual_tol = tol * torch.dot(b, b)
    # residual_tol = tol * rdotr
    if rdotr < residual_tol or rdotr < atol:
        return x, 1
    for i in range(nsteps):
        # To compute Avp
        # h_1 = Hvp_vec(grad_vec=grad_x, params=y_params, vec=lr_x * p, retain_graph=True)
        h_1 = Hvp_vec(grad_vec=grad_x, params=y_params,
                      vec=lr_x * p, retain_graph=True).mul_(lr_y)
        # h_1.mul_(lr_y)
        # lr_y * D_yx * b
        # h_2 = Hvp_vec(grad_vec=grad_y, params=x_params, vec=lr_y * h_1, retain_graph=True)
        h_2 = Hvp_vec(grad_vec=grad_y, params=x_params,
                      vec=h_1, retain_graph=True).mul_(lr_x)
        # h_2.mul_(lr_x)
        # lr_x * D_xy * lr_y * D_yx * b
        Avp_ = p + h_2

        alpha = rdotr / torch.dot(p, Avp_)
        x.data.add_(alpha * p)
        r.data.add_(- alpha * Avp_)
        new_rdotr = torch.dot(r, r)
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
        if rdotr < residual_tol or rdotr < atol:
            break
    if i > 100:
        warnings.warn('CG iter num: %d' % (i + 1))
    return x, i + 1


def conjugate_gradient_precondition(grad_xf, grad_xg,
                                    grad_yf, grad_yg,
                                    x_params, y_params, b,
                                    lr_x, lr_y, x0=None,
                                    nsteps=None,
                                    tol=1e-10, atol=1e-16,
                                    device=torch.device('cpu')):
    '''
    Compute (A^TA)^{-1} b
    where A is given by:
        A   = I - \eta_x D_{xy}f \eta_y D_{yx}g
        A^T = I - D_{xy}g \eta_y D_{yx}f \eta_x
    :param grad_xf: for computing D_{yx}f
    :param grad_xg: for computing D_{yx}g
    :param grad_yf: for computing D_{xy}f
    :param grad_yg: for computing D_{xy}g
    :param x_params:
    :param y_params:
    :param b:
    :param lr_x:
    :param lr_y:
    :param x0:
    :param nsteps:
    :param tol:
    :param atol:
    :param device:
    :return: (A^TA)^{-1} b
    '''
    if x0 is None:
        x = torch.zeros(b.shape[0], device=device)
        r = b.clone()
    else:
        v1 = Matvec(grad_xg=grad_xg, grad_yf=grad_yf,
                    x_params=x_params, y_params=y_params,
                    vec=x0, lr_x=lr_x, lr_y=lr_y,
                    transpose=False)
        v2 = Matvec(grad_xg=grad_xf, grad_yf=grad_yg,
                    x_params=x_params, y_params=y_params,
                    vec=v1, lr_x=lr_x, lr_y=lr_y,
                    transpose=True)
        r = b.clone() - v2
        x = x0

    if nsteps is None:
        nsteps = b.shape[0]
    assert (grad_xf.shape == b.shape), 'CG: hessian vector product shape mismatch'
    assert (grad_xg.shape == b.shape), 'CG: hessian vector product shape mismatch'

    p = r.clone().detach()
    rdotr = torch.dot(r, r)
    residual_tol = tol * torch.dot(b, b)
    if rdotr < residual_tol or rdotr < atol:
        return x, 1
    for i in range(nsteps):
        # Matrix vector product Ap
        v1 = Matvec(grad_xg=grad_xg, grad_yf=grad_yf,
                    x_params=x_params, y_params=y_params,
                    vec=p, lr_x=lr_x, lr_y=lr_x,
                    transpose=False)
        Avp_ = Matvec(grad_xg=grad_xf, grad_yf=grad_yg,
                      x_params=x_params, y_params=y_params,
                      vec=v1, lr_x=lr_x, lr_y=lr_y,
                      transpose=True)

        alpha = rdotr / torch.dot(p, Avp_)
        x.data.add_(alpha * p)
        r.data.add_(- alpha * Avp_)
        new_rdotr = torch.dot(r, r)
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
        if rdotr < residual_tol or rdotr < atol:
            break
    if i > 100:
        warnings.warn('CG iter num: %d' % (i + 1))
    return x, i + 1


def gd_solver(grad_x, grad_y,
              x_params, y_params,
              b, x=None, iter_num=1,
              lr_x=1.0, lr_y=1.0,
              rho=0.9, beta=0.1,
              device=torch.device('cpu')):
    '''
    slove inversion by gradient descent
    '''
    lr_x = lr_x.sqrt()
    if x is None:
        x = torch.zeros(b.shape[0], device=device)
    for i in range(iter_num):
        h1 = Hvp_vec(grad_vec=grad_x, params=y_params, vec=lr_x * x, retain_graph=True).mul_(lr_y)
        h2 = Hvp_vec(grad_vec=grad_y, params=x_params, vec=h1, retain_graph=True).mul_(lr_x)
        delta_x = x + h2 - b
        x = rho * x - beta * delta_x
    return x, 1


def zero_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.detach()
            p.grad.zero_()


def update_params(params, delta):
    if delta == 0.0:
        return
    index = 0
    for p in params:
        p.data.add_(delta[index: index + p.numel()].reshape(p.shape))
        index += p.numel()
    if index != delta.numel():
        raise ValueError('CG size mismatch')


def compute_grad(loss, params):
    '''
    Compute gradient vector w.r.t. params
    :param loss: objective function
    :param params: parameters
    :return:
    '''
    grads = autograd.grad(loss, params, create_graph=True, retain_graph=True)
    grad_vec = torch.cat([g.contiguous().view(-1) for g in grads])
    return grad_vec

