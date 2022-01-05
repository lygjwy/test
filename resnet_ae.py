import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ResizeConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x
    
    
class BasicBlockEnc(nn.Module):
    
    def __init__(self, in_planes, stride=1):
        # in_planes 64
        super().__init__()
        
        planes = in_planes * stride
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
    
class BasicBlockDec(nn.Module):
    
    def __init__(self, in_planes, stride=1):
        super().__init__()
        
        planes = int(in_planes / stride)
        
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        
        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )
            
    def forward(self, x):
        out = F.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = F.relu(out)
        
        return out
    
    
class ResNetEnc(nn.Module):
    
    def __init__(self, num_blocks):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        
    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # x = F.adaptive_avg_pool2d(x, 1)
        # adopt original resnet
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)  # shape N * 512
        
        return out


class ResNetDec(nn.Module):
    
    def __init__(self, num_blocks):
        super().__init__()
        self.in_planes = 512
        
        # self.linear = nn.Linear(512, z_dim)
        
        self.layer4 = self._make_layer(256, num_blocks[3], stride=2)
        self.layer3 = self._make_layer(128, num_blocks[2], stride=2)
        self.layer2 = self._make_layer(64, num_blocks[1], stride=2)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, 3, kernel_size=3, scale_factor=2)
        
    
    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        
        return nn.Sequential(*layers)
    
    def forward(self, z):
        # x = self.linear(z)
        x = z.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=2)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = torch.sigmoid(self.conv1(x))
        x = x.view(x.size(0), 3, 32, 32)
        return x
    

class LinearDec(nn.Module):
    # linear decoder arch
    
    def __init__(self):
        super(LinearDec, self).__init__()
        self.fc = nn.Linear(512, 3 * 32 * 32)

    def forward(self, z):
        x = F.sigmoid(self.fc(z))
        x = x.view(x.size(0), 3, 32, 32)
        return x

    
class ResnetAE(nn.Module):
    
    def __init__(self, name, num_classes, task):
        super().__init__()
        self.task = task
        if name == 'resnet18':
            self.encoder = ResNetEnc([2, 2, 2, 2])
            self.classifier = nn.Linear(512, num_classes)
            self.decoder = ResNetDec([2, 2, 2, 2])
        else:
            raise RuntimeError('<--- not supported arch {}'.format(name))
        
        
    def forward(self, x):
        # feature extractor
        z = self.encoder(x)
        #  classification head
        logit = self.classifier(z)
        #  reconstruction head
        reconstruction = self.decoder(z)
        if self.task == 'classify':
            return logit
        elif self.task == 'reconstruct':
            return reconstruction
        else:
            raise RuntimeError('---> invalid task: {}'.format(self.task))


def get_resnet_ae(name, num_classes, task):
    return ResnetAE(name, num_classes, task)