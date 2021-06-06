import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from GANs import GoodDiscriminator

data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'eval': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

datasets = {'train': CIFAR10('../datas/cifar10', transform=data_transforms['train'],
                             download=True, train=True),
            'eval': CIFAR10('../datas/cifar10', transform=data_transforms['eval'],
                            download=True, train=False)}
dataloaders = {'train': DataLoader(datasets['train'], batch_size=64,
                                   shuffle=True, num_workers=4),
               'eval': DataLoader(datasets['eval'], batch_size=64,
                                  shuffle=False, num_workers=4)}
datasizes = {x: len(datasets[x]) for x in ['train', 'eval']}


def train_cls(epoch_num, save_path, device='cpu'):
    D = GoodDiscriminator()
    # for param in D.parameters():
    #     param.requires_grad = False
    num_fcin = D.linear.in_features
    D.linear = nn.Linear(num_fcin, 10)
    D = D.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(D.linear.parameters(), lr=0.01)
    best_acc = 0.0
    for e in range(epoch_num):
        print('Epoch {}/{}'.format(e, epoch_num - 1))
        for phase in ['train', 'eval']:
            if phase == 'train':
                D.train()
            else:
                D.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = D(inputs)
                    _, pred = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_corrects += torch.sum(pred == labels).item()
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / datasizes[phase]
            epoch_acc = running_corrects / datasizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            if phase == 'eval' and epoch_acc > best_acc:
                best_acc = epoch_acc
    print('Best acc: {:.4f}'.format(best_acc))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    state_dict = D.state_dict()
    torch.save({'D': state_dict}, save_path + 'pretrain.pth')
    print('Model is saved at %s' % save_path)


if __name__ == '__main__':
    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
    print(device)
    train_cls(epoch_num=50, save_path='checkpoints/pretrain/', device=device)

