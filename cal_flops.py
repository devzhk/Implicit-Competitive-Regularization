from utils.train_utils import get_model
import torch
from ptflops import get_model_complexity_info


if __name__ == '__main__':
    z_dim = 128
    model_name= 'Resnet'
    model_config = {'image_size': 64,
                    'image_channel': 3,
                    'feature_num': 128,
                    'n_extra_layers': 0,
                    'batchnorm_d': True,
                    'batchnorm_g': True}
    with torch.cuda.device(0):
        D, G = get_model(model_name=model_name, z_dim=z_dim, configs=model_config)
        macsD, paramsD = get_model_complexity_info(D,
                                                   (model_config['image_channel'],
                                                    model_config['image_size'],
                                                    model_config['image_size']),
                                                   as_strings=True,
                                                   print_per_layer_stat=True,
                                                   verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macsD))
        print('{:<30}  {:<8}'.format('Number of parameters: ', paramsD))
        macsG, paramsG = get_model_complexity_info(G,
                                                   (z_dim,),
                                                   as_strings=True,
                                                   print_per_layer_stat=True,
                                                   verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macsG))
        print('{:<30}  {:<8}'.format('Number of parameters: ', paramsG))