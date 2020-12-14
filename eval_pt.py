import numpy as np
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.inception import inception_v3
from GANs.models import GoodGenerator, DC_generator, dc_G
from GANs import ResNet32Generator, dcG32
from utils import eval_parser
from metrics.is_biggan import load_inception_net, torch_calculate_frechet_distance, \
    torch_cov, numpy_calculate_frechet_distance


class evalor():
    def __init__(self, G, z_dim, model_dir, log_path, device, batchsize=100, dim=1):
        self.is_flag = False
        self.fid_flag = False
        self.log_path = log_path
        self.device = device
        self.G = G
        self.z_dim = z_dim
        self.dim = dim
        self.batchsize = batchsize
        self.model_dir = model_dir
        self.init_writer()
        self.net = load_inception_net(parallel=False)


    def init_writer(self):
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.f = open(self.log_path + 'metrics.csv', 'w')
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
            is_score = self.get_inception_score(batch_num=500)
            np.set_printoptions(precision=5)
            print('Inception score mean: {}, std: {}'.format(is_score[0], is_score[1]))
            content.update({'is_mean': is_score[0],
                            'is_std': is_score[1]})
        if self.fid_flag:
            fid_score = self.get_fid_score()
            np.set_printoptions(precision=5)
            print('FID score: {}'.format(fid_score))
            content.update({'FID score': fid_score})
        self.writer.writerow(content)
        self.f.flush()

    def eval_metrics(self, begin, end, step,
                     is_flag=True, fid_flag=True,
                     dataname='cifar10'):
        print('%d ==> %d, step: %d' %(begin, end, step))
        self.is_flag = is_flag
        self.fid_flag = fid_flag
        if fid_flag:
            self.load_fid(dataname=dataname)
        with torch.no_grad():
            for i in range(begin, end + step, step):
                self.load_model(model_path=self.model_dir + '%d.pth' % i)
                self.get_metrics(i)
        self.f.close()

    # Hongkai's IS implementation
    def generate_data(self):
        if self.dim == 3:
            z = torch.randn((self.batchsize, self.z_dim, 1, 1), device=self.device)
        else:
            z = torch.randn((self.batchsize, self.z_dim), device=self.device)
        data = self.G(z)
        return data

    def get_inception_score(self, batch_num, splits_num=10):
        # net = inception_v3(pretrained=True, transform_input=False).eval().to(self.device)
        resize_module = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=True).to(
            self.device)
        preds = np.zeros((self.batchsize * batch_num, 1000))
        for e in range(batch_num):
            imgs = resize_module(self.generate_data())
            pred = F.softmax(self.net(imgs), dim=1).data.cpu().numpy()
            preds[e * self.batchsize: e * self.batchsize + self.batchsize] = pred
        split_score = []
        chunk_size = preds.shape[0] // splits_num
        for k in range(splits_num):
            pred_chunk = preds[k * chunk_size: k * chunk_size + chunk_size, :]
            kl_score = pred_chunk * (
                    np.log(pred_chunk) - np.log(np.expand_dims(np.mean(pred_chunk, 0), 0)))
            kl_score = np.mean(np.sum(kl_score, 1))
            split_score.append(np.exp(kl_score))
        return np.mean(split_score), np.std(split_score)

    def load_fid(self, dataname):
        # if dataname == 'cifar10':
        #     stats_path = 'metrics/stats/CIFAR10_inception_moments.npz'
        # elif dataname == 'lsun-bedroom':
        #     stats_path = 'metrics/stats/LSUN-bedroom_inception_moments.npz'
        # elif dataname == 'MNIST':
        stats_path = 'metrics/stats/%s_inception_moments.npz' % dataname
        print('Load stats of %s' % dataname)
        f = np.load(stats_path)
        self.mu_real, self.sigma_real = f['mu'][:], f['sigma'][:]
        self.mu_real = torch.tensor(self.mu_real).float().cuda()
        self.sigma_real = torch.tensor(self.sigma_real).float().cuda()
        f.close()


    def accumulate_activations(self, img_num=50000):
        pool, logits = [], []
        while (torch.cat(logits, 0).shape[0] if len(logits) else 0) < img_num:
            with torch.no_grad():
                images = self.generate_data()
                pool_val, logits_val = self.net(images)
                pool += [pool_val]
                logits += [F.softmax(logits_val, 1)]
        return torch.cat(pool, 0), torch.cat(logits, 0)

    def get_fid_score(self):
        pool, logits = self.accumulate_activations()
        print('Calculating FID...')
        mu, sigma = torch.mean(pool, 0), torch_cov(pool, rowvar=False)
        # fid_score = torch_calculate_frechet_distance(mu, sigma, self.mu_real, self.sigma_real)
        # return fid_score.cpu().numpy()
        fid_score = numpy_calculate_frechet_distance(mu.cpu().numpy(), sigma.cpu().numpy(),
                                                     self.mu_real.cpu().numpy(),
                                                     self.sigma_real.cpu().numpy())
        return fid_score


if __name__ == '__main__':
    parser = eval_parser()
    config = vars(parser.parse_args())
    print(config)
    print('numpy calculation')
    device = torch.device('cuda:0')
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
    elif model == 'dc32':
        G =dcG32(z_dim=z_dim)
    else:
        raise ValueError('No matching generator for %s' % model)
    G.to(device)
    G.eval()
    evalor = evalor(G=G, z_dim=z_dim,
                    model_dir=config['model_dir'],
                    device=device,
                    log_path=config['logdir'],
                    dim=config['dim'])
    evalor.eval_metrics(begin=config['begin'], end=config['end'], step=config['step'],
                        is_flag=config['eval_is'], fid_flag=config['eval_fid'],
                        dataname=config['dataset'])