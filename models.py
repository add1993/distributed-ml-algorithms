import torch as th
import torch.nn as nn
from torch.autograd import Variable
import math

class View(nn.Module):
    def __init__(self,o):
        super(View, self).__init__()
        self.o = o

    def forward(self,x):
        return x.view(-1, self.o)

def num_parameters(model):
    return sum([w.numel() for w in model.parameters()])

class mnistfc(nn.Module):
    def __init__(self, opt):
        super(mnistfc, self).__init__()
        self.name = 'mnsitfc'

        c = 1024
        opt['d'] = 0.25

        self.m = nn.Sequential(
            View(784),
            nn.Dropout(0.2),
            nn.Linear(784,c),
            nn.BatchNorm1d(c),
            nn.ReLU(True),
            nn.Dropout(opt['d']),
            nn.Linear(c,c),
            nn.BatchNorm1d(c),
            nn.ReLU(True),
            nn.Dropout(opt['d']),
            nn.Linear(c,10))

        s = '[%s] Num parameters: %d'%(self.name, num_parameters(self.m))
        print(s)

    def forward(self, x):
        return self.m(x)

class mnistconv(nn.Module):
    def __init__(self, opt):
        super(mnistconv, self).__init__()
        self.name = 'mnistconv'
        opt['d'] = 0.5

        def convbn(ci,co,ksz,psz,p):
            return nn.Sequential(
                nn.Conv2d(ci,co,ksz),
                nn.BatchNorm2d(co),
                nn.ReLU(True),
                nn.MaxPool2d(psz,stride=psz),
                nn.Dropout(p))

        self.m = nn.Sequential(
            convbn(1,20,5,3,opt['d']),
            convbn(20,50,5,2,opt['d']),
            View(50*2*2),
            nn.Linear(50*2*2, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(True),
            nn.Dropout(opt['d']),
            nn.Linear(500,10))

        s = '[%s] Num parameters: %d'%(self.name, num_parameters(self.m))
        print(s)

    def forward(self, x):
        return self.m(x)

class allcnn(nn.Module):
    def __init__(self, opt = {'d':0.5}, c1=96, c2= 192):
        super(allcnn, self).__init__()
        self.name = 'allcnn'
        opt['d'] = 0.25

        def convbn(ci,co,ksz,s=1,pz=0):
            return nn.Sequential(
                nn.Conv2d(ci,co,ksz,stride=s,padding=pz),
                nn.BatchNorm2d(co),
                nn.ReLU(True))

        self.m = nn.Sequential(
            nn.Dropout(0.2),
            convbn(3,c1,3,1,1),
            convbn(c1,c1,3,1,1),
            convbn(c1,c1,3,2,1),
            nn.Dropout(opt['d']),
            convbn(c1,c2,3,1,1),
            convbn(c2,c2,3,1,1),
            convbn(c2,c2,3,2,1),
            nn.Dropout(opt['d']),
            convbn(c2,c2,3,1,1),
            convbn(c2,c2,3,1,1),
            convbn(c2,10,1,1),
            nn.AvgPool2d(8),
            View(10))

        s = '[%s] Num parameters: %d'%(self.name, num_parameters(self.m))
        print(s)

    def forward(self, x):
        return self.m(x)

'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
#import torch.nn.functional as F
#
#class BasicBlock(nn.Module):
#    expansion = 1
#
#    def __init__(self, in_planes, planes, stride=1):
#        super(BasicBlock, self).__init__()
#        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#        self.bn1 = nn.BatchNorm2d(planes)
#        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#        self.bn2 = nn.BatchNorm2d(planes)
#
#        self.shortcut = nn.Sequential()
#        if stride != 1 or in_planes != self.expansion*planes:
#            self.shortcut = nn.Sequential(
#                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                nn.BatchNorm2d(self.expansion*planes)
#            )
#
#    def forward(self, x):
#        out = F.relu(self.bn1(self.conv1(x)))
#        out = self.bn2(self.conv2(out))
#        out += self.shortcut(x)
#        out = F.relu(out)
#        return out
#
#
#class Bottleneck(nn.Module):
#    expansion = 4
#
#    def __init__(self, in_planes, planes, stride=1):
#        super(Bottleneck, self).__init__()
#        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#        self.bn1 = nn.BatchNorm2d(planes)
#        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#        self.bn2 = nn.BatchNorm2d(planes)
#        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
#        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
#
#        self.shortcut = nn.Sequential()
#        if stride != 1 or in_planes != self.expansion*planes:
#            self.shortcut = nn.Sequential(
#                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                nn.BatchNorm2d(self.expansion*planes)
#            )
#
#    def forward(self, x):
#        out = F.relu(self.bn1(self.conv1(x)))
#        out = F.relu(self.bn2(self.conv2(out)))
#        out = self.bn3(self.conv3(out))
#        out += self.shortcut(x)
#        out = F.relu(out)
#        return out
#
#
#class ResNet(nn.Module):
#    def __init__(self, block, num_blocks, num_classes=10):
#        super(ResNet, self).__init__()
#        self.in_planes = 64
#
#        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#        self.bn1 = nn.BatchNorm2d(64)
#        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#        self.linear = nn.Linear(512*block.expansion, num_classes)
#
#    def _make_layer(self, block, planes, num_blocks, stride):
#        strides = [stride] + [1]*(num_blocks-1)
#        layers = []
#        for stride in strides:
#            layers.append(block(self.in_planes, planes, stride))
#            self.in_planes = planes * block.expansion
#        return nn.Sequential(*layers)
#
#    def forward(self, x):
#        out = F.relu(self.bn1(self.conv1(x)))
#        out = self.layer1(out)
#        out = self.layer2(out)
#        out = self.layer3(out)
#        out = self.layer4(out)
#        out = F.avg_pool2d(out, 4)
#        out = out.view(out.size(0), -1)
#        out = self.linear(out)
#        return out
#
#
#def ResNet18(num_classes=10):
#    return ResNet(BasicBlock, [2,2,2,2], num_classes)
#
#def ResNet34(num_classes=10):
#    return ResNet(BasicBlock, [3,4,6,3], num_classes)
#
#def ResNet50(num_classes=10):
#    return ResNet(Bottleneck, [3,4,6,3], num_classes)
#
#def ResNet101(num_classes=10):
#    return ResNet(Bottleneck, [3,4,23,3], num_classes)
#
#def ResNet152(num_classes=10):
#    return ResNet(Bottleneck, [3,8,36,3], num_classes)


# def test():
#     net = ResNet18()
#     y = net(torch.randn(1,3,32,32))
#     print(y)
#     print(y.size())

#test()
