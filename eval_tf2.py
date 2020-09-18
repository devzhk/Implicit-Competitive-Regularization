import os
import torch
import csv
import numpy as np
from metrics.InceptionScore_tf2 import get_inception_score
from GANs.models import GoodGenerator

BATCH_SIZE = 100
N_CHANNEL = 3
RESOLUTION = 32
NUM_SAMPLES = 50000


def cal_inception_score(G, device, z_dim):
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
        print('loading model from %s' % model_path)
        chkpoint = torch.load(model_path)
        self.G.load_state_dict(chkpoint['G'])

    def get_metrics(self, count):
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

    def eval_metrics(self, begin, end, step, is_flag=True, fid_flag=False):
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
    G = GoodGenerator().to(device)
    model_path = 'checkpoints/ACGD64/ACGD-dc0.010_'
    logdir = 'eval_results/tfACGD64/'
    evalor = Evalor(G=G, z_dim=128, dataset='cifar10',
                    model_dir=model_path, log_path=logdir,
                    device=device)
    evalor.eval_metrics(begin=300000, end=465000, step=5000)
