<h1 align="center">Implicit competitive regularization in GANs</h1>

This code contains experiments for paper: 'Implicit competitive regularization in GANs': https://arxiv.org/abs/1910.05852

**Warning**: This implementation is only for zero sum game setting because it relies on conjugate gradient method to solve matrix inversion efficiently, which requires the matrix to be positive definite. If you are using competitive gradient descent (CGD) algorithm for non-zero sum games, please check more details in CGD paper https://arxiv.org/abs/1905.12103. For example, GMRES (the generalized minimal residual) algorithm can be a solver for non-zero sum setting. 



## How to use new optimizer(ACGD) in our paper
Package 'optims' contains the original Compeititive Gradient Descent (BCGD), and the Adaptive Competitive Gradient Descent (ACGD). 
**It's important to force cudnn to benchmark and pick the best algo.**

```python
import torch
torch.backends.cudnn.benchmark = True
from optims import ACGD
device = torch.device('cuda:0')
lr = 0.0001
G = Generator()
D = Discriminator()
optimizer = ACGD(max_params=G.parameters(), min_params=D.parameters(), lr_max=lr, lr_min=lr, device=device)
# max_parems is maximizing the objective function while the min_params is trying to minimizing it. 
# BCGD(max_params=G.parameters(), min_params=D.parameters(), lr_max=lr, lr_min=lr, device=device)
# ACGD: Adaptive CGD;
for img in dataloader:
    d_real = D(img)
    z = torch.randn((batch_size, z_dim), device=device)
    d_fake = D(G(z))
    loss = criterion(d_real, d_fake)
    optimizer.zero_grad()
    optimizer.step(loss=loss)
```


## Citation
Please cite the following paper if you find this code useful. Thanks!
```
@misc{schfer2019implicit,
    title={Implicit competitive regularization in GANs},
    author={Florian Schäfer and Hongkai Zheng and Anima Anandkumar},
    year={2019},
    eprint={1910.05852},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
