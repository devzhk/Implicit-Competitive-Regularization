import os
from train_utils import get_data
import torchvision.utils as vutils
from torch.utils.data import DataLoader


def generate(dataname, path, save_path, batch_size=64, device='cpu'):
    dataset = get_data(dataname=dataname, path='../datas/%s' % path)
    real_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,
                             num_workers=4)
    real_set = next(iter(real_loader))
    real_set = real_set[0].to(device)
    if not os.path.exists('figs/%s' % save_path):
        os.makedirs('figs/%s' % save_path)
    vutils.save_image(real_set, 'figs/%s/%s.png' % (save_path, dataname), normalize=True)


# def gen_from_model(chk, model_name, batchsize=64, device='cpu'):


if __name__ == '__main__':
    generate(dataname='CIFAR10', path='cifar10', save_path='cifar10', batch_size=64)