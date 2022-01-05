from __future__ import print_function
import numpy as np
import argparse
import matplotlib.pyplot as plt

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from corrupt_named_dataset_with_meta import CorruptNamedDatasetWithMeta

from ae import get_ae
from named_dataset_with_meta import NamedDatasetWithMeta
from resnet import get_resnet18
from resnet_ae import get_resnet_ae


# scheduler
def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


parser = argparse.ArgumentParser(description='AE')
parser.add_argument('--task', type=str, default='classify', required=True)
parser.add_argument('--mode', type=str, default='normal', required=True)
parser.add_argument('--preprocess', type=str, default='without_normalize')
parser.add_argument('--arch', type=str, default='resnet_ae', required=True)
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--loss_fn', type=str, default='mse')
parser.add_argument('--reduction', type=str, default='sum')
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--output_dir', type=str, default='./results')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=36, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--gpu_idx', type=int, default=0)

args = parser.parse_args()


torch.manual_seed(args.seed)

device = torch.device("cuda:" + str(args.gpu_idx) if torch.cuda.is_available() else "cpu")

kwargs = {'num_workers': 8, 'pin_memory': True}


if args.task == 'classify':
    if args.preprocess == 'normalize':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    elif args.preprocess == 'without_normalize':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor()
        ])
    else:
        raise RuntimeError('---> invalid preprocess: {}'.format(args.preprocess))
elif args.task == 'reconstruct':
    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])
else:
    raise RuntimeError('---> not implemented task: {}'.format())

if args.task == 'classify' and args.preprocess == 'normalize':
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
else:
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

if args.mode == 'normal':
    train_loader = torch.utils.data.DataLoader(
        NamedDatasetWithMeta('/home/iip/datasets', 'cifar10', split='train', transform=train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        NamedDatasetWithMeta('/home/iip/datasets', 'cifar10', split='test', transform=test_transform),
        batch_size=args.batch_size,
        shuffle=False,
        **kwargs
    )
elif args.mode == 'corrupt':
    train_loader = torch.utils.data.DataLoader(
        CorruptNamedDatasetWithMeta('/home/iip/datasets', 'cifar10', 'gaussian_noise', severity=5, split='train', transform=train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        CorruptNamedDatasetWithMeta('/home/iip/datasets', 'cifar10', 'gaussian_noise', severity=5, split='test', transform=test_transform),
        batch_size=args.batch_size,
        shuffle=False,
        **kwargs
    )
else:
    raise RuntimeError('---> invalid mode: {}'.format(args.mode))


arch = args.arch
if arch == 'ae':
    model = get_ae('resnet18').to(device)
elif arch == 'resnet18':
    model = get_resnet18(num_classes=10, include_top=True)
elif arch == 'resnet18_ae':
    model = get_resnet_ae('resnet18', 10, args.task)
else:
    raise RuntimeError('---> invalid arch: {}'.format(arch))
model.to(device)


if args.optimizer == 'adam':
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr, weight_decay=5e-4)
elif args.optimizer == 'sgd':
    lr = 0.1
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True
    )
else:
    raise RuntimeError('---> invalid optimizer: {}'.format(args.optimizer))
    
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        args.epochs * len(train_loader),
        1,
        1e-6 / lr
    )
)

def reconstruction_loss_function(recon_x, x):
    recon_loss =  F.mse_loss(recon_x.view(-1, 3072), x.view(-1, 3072), reduction='sum')
    return recon_loss

def classify_loss_function(logit, target):
    cla_loss = F.cross_entropy(logit, target)
    return cla_loss


average_train_loss_list = []
average_test_loss_list = []

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def visualize():
    x = list(range(args.epochs))
    l1 = plt.plot(x, average_train_loss_list, 'r--', label='train loss')
    l2 = plt.plot(x, average_test_loss_list, 'g--', label='test loss')

    name = '-'.join([args.loss_fn, args.reduction, str(args.epochs)])
    plt.title(name)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(args.output_dir + '/' + name + '.png')
    # save loss_lists to local
    with open(args.output_dir + '/' + name + '.txt', 'w') as f:
        f.write(str(average_train_loss_list)+'\n')
        f.write(str(average_test_loss_list))

def train(epoch):
    model.train()
    total = 0
    correct = 0
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        
        if args.task == 'reconstruct':
            if args.mode == 'normal':
                data, _ = data
                data = data.to(device)
                recon_data = model(data)
            elif args.mode == 'corrupt':
                corrupt_data, data, _ = data
                data = data.to(device)
                corrupt_data = corrupt_data.to(device)
                recon_data = model(corrupt_data)
            else:
                raise RuntimeError('not implemented mode: {}'.format(args.mode))
        elif args.task == 'classify':
            data, target = data
            total += len(target)
            data = data.to(device)
            target = target.to(device)
            logit = model(data)
            
            pred = logit.data.max(dim=1)[1]
            correct += pred.eq(target.data).sum().item()
        else:
            raise RuntimeError('---> invalid task: {}'.format(args.task))
        
        # data = data.to(device)
        optimizer.zero_grad()
        if args.task == 'classify':
            loss = classify_loss_function(logit, target)
        elif args.task == 'reconstruct':
            loss = reconstruction_loss_function(recon_data, data)
        else:
            raise RuntimeError('---> invalid task: {}'.format(args.task))
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        scheduler.step()
        
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader),
        #         loss.item()))

    # batch average 
    average_train_loss = train_loss / len(train_loader)
    average_train_loss_list.append(average_train_loss)
    if args.task == 'classify':
        print('---> train loss: {:.4f} train cla: {:.4f}'.format(average_train_loss, 100 * correct / total))
    elif args.task == 'reconstruct':
        print('---> train loss: {:.4f}'.format(average_train_loss))
    else:
        raise RuntimeError('---> invalid task: {}'.format(args.task))


def test(epoch):
    model.eval()
    total = 0
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            
            if args.task == 'reconstruct':
                if args.mode == 'normal':
                    data, _ = sample
                    data = data.to(device)
                    rec_data = model(data)
                elif args.mode == 'corrupt':
                    cor_data, data, _ = sample
                    cor_data, data = cor_data.to(device), data.to(device)
                    rec_data = model(cor_data)
                else:
                    raise RuntimeError('not implemented mode: {}'.format(args.mode))
                
                test_loss += reconstruction_loss_function(rec_data, data).item()
                #  unnormalize to visualize
                # unorm = UnNormalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))
            
                if i == 0:
                    n = min(data.size(0), 8)
                    rec_data = rec_data.view(args.batch_size, 3, 32, 32)
                    # unorm_data = torch.stack([unorm(data_item) for data_item in data[:n]])
                    # unorm_recon = torch.stack([unorm(data_item) for data_item in recon_data[:n]])
                    if args.mode == 'normal':
                        comparison = torch.cat([data[:n], rec_data[:n]])
                    elif args.mode == 'corrupt':
                        comparison = torch.cat([data[:n], cor_data[:n], rec_data[:n]])
                    else:
                        raise RuntimeError('not implemented mode: {}'.format(args.mode))
                    
                    save_image(comparison.cpu(),
                            args.output_dir + '/reconstruction_' + args.loss_fn + '_' + str(epoch) + '.png', nrow=n)
            elif args.task == 'classify':
                data, target = sample
                total += len(data)
                
                data, target = data.to(device), target.to(device)
                logit = model(data)
                test_loss += classify_loss_function(logit, target).item()

                pred = logit.data.max(dim=1)[1]
                correct += pred.eq(target.data).sum().item()
            else:
                raise RuntimeError('---> invalid task: {}'.format(args.task))
            
    # average_test_loss = test_loss / len(test_loader.dataset)
    average_test_loss = test_loss / len(test_loader)
    average_test_loss_list.append(average_test_loss)
    if args.task == 'classify':
        print('---> test loss: {:.4f} test cla: {:.4f}'.format(average_test_loss, 100 * correct / total))
    elif args.task == 'reconstruct':
        print('---> test loss: {:.4f}'.format(average_test_loss))
    else:
        raise RuntimeError('---> invalid task: {}'.format(args.task))


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        print('---> Epoch: {}'.format(epoch))
        train(epoch)
        test(epoch)
    
    visualize()
    