import os
import csv
import numpy as np
import torch

from GANs.models import GoodGenerator, DC_generator
from metrics.cifar10 import cal_inception_score, cal_fid_score
from metrics.lsun_bedroom import cal_fid_score as lsun_fid_score
from utils import eval_parser


class Evalor():
    def __init__(self, G, z_dim, dataset,
                 model_dir,
                 log_path, device):
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
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.f = open(self.log_path + '%s_metrics.csv' % self.dataset, 'w')
        fieldnames = ['iter',
                      'is_mean', 'is_std',
                      'FID score']
        self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
        self.writer.writeheader()

    def load_model(self, model_path):
        chkpoint = torch.load(model_path)
        self.G.load_state_dict(chkpoint['G'])
        print('loading model from %s' % model_path)

    def get_metrics(self, count):
        print('===Iter %d===' % count)
        content = {'iter': count}
        if self.is_flag:
            is_score = cal_inception_score(G=self.G, device=self.device, z_dim=self.z_dim)
            np.set_printoptions(precision=5)
            print('Inception score mean: {}, std: {}'.format(is_score[0], is_score[1]))
            content.update({'is_mean': is_score[0],
                            'is_std': is_score[1]})
        if self.fid_flag:
            if self.dataset == 'lsun-bedroom':
                fid_score = lsun_fid_score(G=self.G, device=device, z_dim=self.z_dim)
            elif self.dataset == 'cifar10':
                fid_score = cal_fid_score(G=self.G, device=self.device, z_dim=self.z_dim)
            np.set_printoptions(precision=5)
            print('FID score : {}'.format(fid_score))
            content.update({'FID score': fid_score})
        self.writer.writerow(content)
        self.f.flush()

    def eval_metrics(self, begin, end, step, is_flag=True, fid_flag=True):
        print('%d ==> %d, step: %d' %(begin, end, step))
        self.is_flag = is_flag
        self.fid_flag = fid_flag
        for i in range(begin, end + step, step):
            self.load_model(model_path=self.model_dir + '%d.pth' % i)
            self.get_metrics(i)
        self.f.close()


if __name__ == '__main__':
    parser = eval_parser()
    config = vars(parser.parse_args())
    print(config)
    device = torch.device('cuda:0')
    if config['model'] == 'dc':
        G = GoodGenerator().to(device)
    elif config['model'] == 'DCGAN':
        G = DC_generator(z_dim=config['z_dim']).to(device)
    # z_dim = config['z_dim']
    # model_dir = '/checkpoints/0.00000CIFAR-W-0.0001/ACGD-0.00010_'
    # logdir = '/eval_results/CIFAR-WN-ACGD/'
    evalor = Evalor(G=G, z_dim=config['z_dim'], dataset=config['dataset'],
                    model_dir=config['model_dir'], device=device, log_path=config['logdir'])
    evalor.eval_metrics(begin=config['begin'], end=config['end'], step=config['step'],
                        is_flag=config['eval_is'], fid_flag=config['eval_fid'])
