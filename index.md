# Stabilize GAN training

**Why it matters**: Generative Adversarial Networks (GANs) is one of the most successful deep generative models that has been widely applied in various areas. However, different from training other generative models, GANs update the generator and the discriminator in a minmax optimization form, which makes it more of an art than a science to train. Finding a stable GAN training algorithm remains a big challenge. 

**Related works**: In general, it is hard to come up with an algorithm that guarantees the global convergence because both discriminator and generator are non-convex functions of their parameters. But some progress has been made by simplifying the problem setting such as linear generator, Gaussian data. Aside from it, there are many papers proposing regularization techniques to stabilize training, which could fail badly in complex and multimodal domains. Another line of research propose to model GAN training from a game-theoretic perspective. These techniques yield training procedures that provably converge to some kind of approximate Nash equilibrium, but do so using unreasonably large resource constraints. 

**What we have now**
Our previous work provides empirical evidence of how GAN training can be stabilized by utilizing the interactions between discriminator and generator. [Implicit Competitive Regularization in GANs]([Implicit competitive regularization in GANs ICML 2020](https://proceedings.mlr.press/v119/schaefer20a.html))
We' have developed a practical optimization algorithm - Adaptive Competitive Gradient Descent (ACGD). [Google Colab: train a GAN using ACGD](https://colab.research.google.com/drive/1-52aReaBAPNBtq2NcHxKkVIbdVXdyqtH?usp=sharing)
