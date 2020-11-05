'''
Code derived from https://github.com/tsc2017/Inception-Score
Arguments:
    --model: generator architecture
    --dataset: the dataset to compare with
    --z_dim: latent dimensionality
    --dim: dim of latent tensor
    --begin: the iteration number to begin from
    --end: the iteration number to end with
    --step: the step of iteration
    --model_dir: the directory of model checkpoints
    --log_dir: the directory to save the results
    --eval_is: True or False to evaluate IS
    --eval_fid: True or False to evaluate FID
Outputs:
csv file that records the scores of checkpoints
'''
import os
import torch
import csv
import numpy as np
from metrics.InceptionScore_tf2 import get_inception_score
from GANs.models import GoodGenerator, DC_generator, dc_G
from GANs import ResNet32Generator
from utils import eval_parser

BATCH_SIZE = 100
N_CHANNEL = 3
RESOLUTION = 32
NUM_SAMPLES = 50000


def cal_inception_score(G, device, z_dim):
    """
    Calculate the proposal score.

    Args:
        G: (array): write your description
        device: (todo): write your description
        z_dim: (int): write your description
    """
    all_samples = []
    samples = torch.randn(NUM_SAMPLES, z_dim)
    for i in range(0, NUM_SAMPLES, BATCH_SIZE):
        samples_100 = samples[i:i + BATCH_SIZE]
        samples_100 = samples_100.to(device=device)
        all_samples.append(G(samples_100).cpu().data.numpy())

    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = ((all_samples + 1)/ 2 * 255).astype(np.uint8)
    all_samples = all_samples.reshape((-1, N_CHANNEL, RESOLUTION, RESOLUTION))
    return get_inception_score(all_samples)


class Evalor(object):
    def __init__(self, G, z_dim, dataset,
                 model_dir,
                 log_path, device):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            G: (int): write your description
            z_dim: (int): write your description
            dataset: (todo): write your description
            model_dir: (str): write your description
            log_path: (str): write your description
            device: (todo): write your description
        """
        self.is_flag = False
        self.fid_flag = False
        self.log_path = log_path
        self.device = device
        self.G = G
        self.z_dim = z_dim
        self.model_dir = model_dir
        self.dataset = dataset
        self.init_writer()

    def init_writer(self):
        """
        Init csv file

        Args:
            self: (todo): write your description
        """
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.f = open(self.log_path + '%s_metrics.csv' % self.dataset, 'w')
        fieldnames = ['iter',
                      'is_mean', 'is_std',
                      'FID score']
        self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
        self.writer.writeheader()

    def load_model(self, model_path):
        """
        Load the model from disk.

        Args:
            self: (str): write your description
            model_path: (str): write your description
        """
        print('loading model from %s' % model_path)
        chkpoint = torch.load(model_path)
        self.G.load_state_dict(chkpoint['G'])

    def get_metrics(self, count):
        """
        Get metrics.

        Args:
            self: (todo): write your description
            count: (str): write your description
        """
        print('===Iter %d===' % count)
        content = {'iter': count}
        if self.is_flag:
            is_score = cal_inception_score(G=self.G, device=self.device, z_dim=self.z_dim)
            np.set_printoptions(precision=5)
            print('Inception score mean: {}, std: {}'.format(is_score[0], is_score[1]))
            content.update({'is_mean': is_score[0],
                            'is_std': is_score[1]})
        # if self.fid_flag:
            # if self.dataset == 'lsun-bedroom':
            #     fid_score = lsun_fid_score(G=self.G, device=device, z_dim=self.z_dim)
            # elif self.dataset == 'cifar10':
            #     fid_score = cal_fid_score(G=self.G, device=self.device, z_dim=self.z_dim)
            # np.set_printoptions(precision=5)
            # print('FID score : {}'.format(fid_score))
            # content.update({'FID score': fid_score})
        self.writer.writerow(content)
        self.f.flush()

    def eval_metrics(self, begin, end, step, is_flag=True, fid_flag=False, dataname='CIFAR10'):
        """
        Evaluate the model

        Args:
            self: (todo): write your description
            begin: (todo): write your description
            end: (todo): write your description
            step: (todo): write your description
            is_flag: (bool): write your description
            fid_flag: (str): write your description
            dataname: (str): write your description
        """
        print('%d ==> %d, step: %d' % (begin, end, step))
        self.is_flag = is_flag
        self.fid_flag = fid_flag
        for i in range(begin, end + step, step):
            self.load_model(model_path=self.model_dir + '%d.pth' % i)
            self.get_metrics(i)
        self.f.close()


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    parser = eval_parser()
    config = vars(parser.parse_args())
    print(config)
    model = config['model']
    z_dim = config['z_dim']
    if model == 'dc':
        G = GoodGenerator()
    elif model == 'ResGAN':
        G = ResNet32Generator(z_dim=z_dim, num_filters=128, batchnorm=True)
    elif model == 'DCGAN':
        G = DC_generator(z_dim=z_dim)
    elif model == 'mnist':
        G = dc_G(z_dim=z_dim)
    G.to(device)
    G.eval()
    evalor = Evalor(G=G, z_dim=128, dataset='cifar10',
                    model_dir=config['model_dir'],
                    log_path=config['logdir'],
                    device=device)
    evalor.eval_metrics(begin=config['begin'], end=config['end'], step=config['step'],
                        is_flag=config['eval_is'], fid_flag=config['eval_fid'],
                        dataname=config['dataset'])
