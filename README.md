<h1 align="center">Implicit competitive regularization in GANs</h1>

This code contains experiments for paper: 'Implicit competitive regularization in GANs': https://arxiv.org/abs/1910.05852

Quick demo for experiments : `train.ipynb`
Repro code for paper: `VisionData.py`, `wgan_gp.py`

## How to use new optimizer(ACGD) in our paper
Copy `optimizers.py` and `cgd_utils.py` to your directory. 
**It's important to force cudnn to benchmark and pick the best algo.**
```python
import torch
torch.backends.cudnn.benchmark = True
from optims import ACGD
device = torch.device('cuda:0')
lr = 0.0001
G = Generator()
D = Discriminator()
optimizer = ACGD(max_params=G.parameters(), min_params=D.parameters(), lr=lr, device=device)
# BCGD(max_params=G.parameters(), min_params=D.parameters(), lr=lr, device=device)
# ACGD: Adaptive learning rates CGD;
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
