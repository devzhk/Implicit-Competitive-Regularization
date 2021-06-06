import pandas as pd
import torch

from torch.utils.data import DataLoader

from optims import OCGD
from utils.train_utils import get_data, save_checkpoint, get_model
from utils.losses import get_loss


# seed = torch.randint(0, 1000000, (1,))
seed = 2020
torch.manual_seed(seed=seed)
print('random seed : %d' % seed)


def train_ocgd(epoch_num=10, optim_type='BCGD2',
               startPoint=None, logdir='test',
               update_min=True,
               z_dim=128, batchsize=64,
               loss_name='WGAN', model_name='dc',
               data_path='None', dataname='cifar10',
               device='cpu', gpu_num=1, collect_info=False):
    lr_d = 0.01
    lr_g = 0.01
    dataset = get_data(dataname=dataname, path='../datas/%s' % data_path)
    dataloader = DataLoader(dataset=dataset, batch_size=batchsize, shuffle=True,
                            num_workers=4)
    D, G = get_model(model_name=model_name, z_dim=z_dim)
    D.to(device)
    G.to(device)
    if startPoint is not None:
        chk = torch.load(startPoint)
        D.load_state_dict(chk['D'])
        G.load_state_dict(chk['G'])
        print('Start from %s' % startPoint)

    optimizer = OCGD(max_params=G.parameters(), min_params=D.parameters(),
                     udpate_min=update_min, device=device)
    loss_list = []
    count = 0
    for e in range(epoch_num):
        for real_x in dataloader:
            real_x = real_x[0].to(device)
            d_real = D(real_x)
            z = torch.randn((real_x.shape[0], z_dim), device=device)
            fake_x = G(z)
            d_fake = D(fake_x)
            D_loss = get_loss(name=loss_name, g_loss=False, d_real=d_real, d_fake=d_fake)
            optimizer.zero_grad()
            optimizer.step(loss=D_loss)
            if count % 100 == 0:
                print('Iter %d, Loss: %.5f' % (count, D_loss.item()))
                loss_list.append(D_loss.item())
            count += 1
        print('epoch{%d/%d}' %(e, epoch_num))
    name = 'overtrainD.pth' if update_min else 'overtrainG.pth'
    save_checkpoint(path=logdir, name=name, D=D, G=G)
    loss_data = pd.DataFrame(loss_list)
    loss_data.to_csv('logs/train_oneside.csv')


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    chk = 'checkpoints/0.00000MNIST-0.0100/SGD-0.01000_9000.pth'
    train_ocgd(epoch_num=10, startPoint=chk,
               z_dim=96, update_min=True,
               data_path='mnist', dataname='MNIST',
               loss_name='JSD', model_name='mnist',
               batchsize=128, device=device)