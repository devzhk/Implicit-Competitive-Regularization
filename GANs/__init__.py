from .models import dc_D, dc_G, \
    GoodDiscriminator, GoodGenerator, GoodDiscriminatord, \
    DC_generator, DC_discriminator, DC_discriminatorW, GoodDiscriminatorbn
from .cifar_resnet import ResNet32Discriminator, ResNet32Generator, ResNetDiscriminator, ResNetGenerator
from .sngans import GoodSNDiscriminator
from .cifar_models import dcD32, dcG32
from .dcgan import DCGAN_D, DCGAN_G