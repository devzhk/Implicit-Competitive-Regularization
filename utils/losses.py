import math
import torch
import torch.nn as nn
from torch import autograd
from torch.nn import functional as F


criterion = nn.BCEWithLogitsLoss()


def get_loss(name, g_loss,
             d_real=None, d_fake=None,
             gp_weight=0, l2_weight=0, D=None):
    if name == 'WGAN':
        loss = - d_fake.mean() if g_loss else d_fake.mean() - d_real.mean()
    elif name == 'JSD':
        if g_loss:
            loss = criterion(d_fake, torch.ones(d_fake.shape, device=d_fake.device))
        else:
            loss = criterion(d_real, torch.ones(d_real.shape, device=d_real.device)) \
                   + criterion(d_fake, torch.zeros(d_fake.shape, device=d_fake.device))
    # TODO add gradient penalty
    if l2_weight != 0:
        l2loss = 0
        for p in D.parameters():
            l2loss += torch.dot(p.view(-1), p.view(-1))
        loss += l2_weight * l2loss
    return loss


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths