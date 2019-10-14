# ICR
code for ICLR 2020: Implicit competitive regularization in GANs

quick start: train.ipynb

Further explanation 
- To train WGAN-GP with Adam: python wgan_gp.py
- To reproduce results on CIFAR10- VisionData.py - train_wgan ()
  - loss_type:  1. 'JSD': original GAN loss 2. 'WGAN' Wasserstein loss
  - dropout: 1. None: disable dropout 2. 0.5: most common dropout
  - d_penalty: L2 penalty with respect to discriminator  
- For experiments on MNIST
  - train_mnist()
  - trainer.train_gd() to train GANs 
  - trainer.train_d() to train discriminator keeping the generator fixed, using Adam
  - trainer.traing() to train generator keeping the discriminator fixed, using Adam
  - trainer. train_ocgd() to train GAN with one side CGD
    - update_D=True: train discriminator keeping generator fixed 
    - update_D=False: train generator keeping discriminator fixed
## How to use CGD
```python
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
