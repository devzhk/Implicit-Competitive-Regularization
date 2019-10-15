# ICR
Code for paper: *Implicit competitive regularization in GANs*: https://arxiv.org/abs/1910.05852

Demo for paper: train.ipynb
    
## How to use CGD
```python
from optimizers import MCGD
device = torch.device('cuda:0')
lr = 0.0001
G = Generator()
D = Discriminator()
optimizer = MCGD(max_params=G, min_params=D, lr=lr, device=device)
for img in dataloader:
    d_real = D(img)
    z = torch.randn((batch_size, z_dim), device=device)
    d_fake = D(G(z))
    loss = criterion(d_real, d_fake)
    optimizer.zero_grad()
    optimizer.step(loss=loss)
```
