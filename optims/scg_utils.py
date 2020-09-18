import torch
import torch.autograd as autograd
from .cgd_utils import Hvp_vec
import warnings

#
# def SHvp_vec(closure, img, x_params, y_params, vec, retain_graph=False):
#     '''
#     return Hessian vector product: \partial^2 f / \partial y \partial x * vec
#     output shape is the same as y_params
#     '''
#     loss = closure(img)
#     grad_x = autograd.grad(loss, x_params, create_graph=True, retain_graph=True)
#     grad_x_vec = torch.cat([g.contiguous().view(-1) for g in grad_x])
#     grad_grad = autograd.grad(grad_x_vec, y_params, grad_outputs=vec,
#                               retain_graph=retain_graph, allow_unused=True)
#     grad_list = []
#     for i, p in enumerate(y_params):
#         if grad_grad[i] is None:
#             grad_list.append(torch.zeros_like(p).view(-1))
#         else:
#             grad_list.append(grad_grad[i].contiguous().view(-1))
#     hvp = torch.cat(grad_list)
#     if torch.isnan(hvp).any():
#         raise ValueError('hvp Nan')
#     return hvp


def Avp(closure, img,
        x_params, y_params,
        vec,
        lr_x=1.0, lr_y=1.0,
        retain_graph=False):
    '''
    Parameters:
        - closure: the closure to define the forward graph
        - img: input data of closure
        - x_params: w.r.t which derivatives are computed
        - y_params: w.r.t which derivatives are computed
        - vec: The "vector" in Matrix vector product
        - lr_x: learning rate vector for x_params
        - lr_y: learning rate vector for y_params
        - retain_graph: If False, the graph used to compute will be freed
    compute matrix vector product :
    (I + lr_x * D_xy * lr_y * D_yx * lr_x) * vec
    return vector
    '''
    loss = closure(img)
    grad_x = autograd.grad(loss, x_params, create_graph=True, retain_graph=True)
    grad_x_vec = torch.cat([g.contiguous().view(-1) for g in grad_x])
    grad_y = autograd.grad(loss, y_params, create_graph=True, retain_graph=True)
    grad_y_vec = torch.cat([g.contiguous().view(-1) for g in grad_y])

    h1 = Hvp_vec(grad_vec=grad_x_vec, params=y_params,
                 vec=lr_x * vec, retain_graph=True).mul_(lr_y)
    # lr_y * D_yx * lr_x * (vec)
    h2 = Hvp_vec(grad_vec=grad_y_vec, params=x_params,
                 vec=h1, retain_graph=retain_graph).mul_(lr_x)
    # lr_x * D_xy * (lr_y * D_yx * lr_x * vec)
    Avp_ = vec + h2
    return Avp_


def sto_conjugate_gradient(closure, dataloader,
                           x_params, y_params, b,
                           lr_x, lr_y,
                           x=None, nsteps=None,
                           tol=1e-10, atol=1e-10,
                           device=torch.device('cpu')):
    '''
    return (I + sqrt(lr_x) * D_xy * lr_y * D_yx * sqrt(lr_x)) ** -1 * b
    '''
    lr_x = lr_x.sqrt()
    if x is None:
        x = torch.zeros(b.shape[0], device=device)
        r = b.clone().detach()
    else:
        img_data = next(dataloader)
        Avx = Avp(closure=closure, img=img_data[0],
                  x_params=x_params, y_params=y_params,
                  vec=x,
                  lr_x=lr_x, lr_y=lr_y,
                  retain_graph=False)
        r = b.clone().detach() - Avx
    if nsteps is None:
        nsteps = b.shape[0]
    p = r.clone().detach()
    rdotr = torch.dot(r, r)
    residual_tol = tol * rdotr
    if rdotr < residual_tol or rdotr < atol:
        return x, 1
    for i in range(nsteps):
        # To compute Avp
        img_data = next(dataloader)
        Avp_ = Avp(closure=closure, img=img_data[0],
                   x_params=x_params, y_params=y_params,
                   vec=p,
                   lr_x=lr_x, lr_y=lr_y,
                   retain_graph=False)
        alpha = rdotr / torch.dot(p, Avp_)
        new_x = torch.add(x, alpha * p)
        r.data.add_(- alpha * Avp_)
        new_rdotr = torch.dot(r, r)
        beta = new_rdotr / rdotr
        p = r + beta * p
        # print('Inner iteration {}: {}'.
        #       format(i+1, rdotr.cpu().numpy()))
        if rdotr < residual_tol or rdotr < atol:
            break
        if new_rdotr > 1.1 * rdotr:
            return x, i + 1
        else:
            rdotr = new_rdotr
            x = new_x
    if i > 99:
        warnings.warn('CG iter num: %d' % (i + 1))
    return x, i + 1
