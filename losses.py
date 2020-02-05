import torch
import torch.nn as nn

criterion = nn.BCEWithLogitsLoss()


def get_loss(name, g_loss,
             d_real=None, d_fake=None,
             gp_weight=0, l2_weight=0):
    if name == 'WGAN':
        loss = - d_fake.mean() if g_loss else d_fake.mean() - d_real.mean()
    elif name == 'JSD':
        if g_loss:
            loss = criterion(d_fake, torch.ones(d_fake.shape, device=d_fake.device))
        else:
            loss = criterion(d_real, torch.ones(d_real.shape, device=d_real.device)) \
                   + criterion(d_fake, torch.zeros(d_fake.shape, device=d_fake.device))
    # TODO add gradient penalty and l2 weight penalty
    return loss