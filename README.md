<h1 align="center">Implicit competitive regularization in GANs</h1>
Code for paper: 'Implicit competitive regularization in GANs': https://arxiv.org/abs/1910.05852

Demo for paper: train.ipynb
    
## How to use CGD
Copy the `optimizers.py` and `cgd_utils.py`
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
## Citation
Please cite the following paper if you found our optimizer useful. Thanks!
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
