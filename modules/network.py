from collections import OrderedDict

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class Net(nn.Module):  # resnet by me
    def __init__(self, arch):
        super(Net, self).__init__()
        # Model Selection
        original_model = models.__dict__[arch](pretrained=False)

        self.features = torch.nn.Sequential(
            *(list(original_model.children())[:-1]))
        self.classifier = nn.Linear(in_features=list(original_model.children())[-1].in_features, out_features=10,
                                    bias=True)
        self.modelName = arch

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


class VGG_Alex(nn.Module):
    def __init__(self, arch):
        # super(Net, self).__init__()
        # Model Selection
        super().__init__()
        num_classes = 10
        original_model = models.__dict__[arch](pretrained=True)

        # 模型使用 预训练特征提取器（卷积层），创建空分类器（全连接层）
        # 不同数据集，使用时resize至同一大小
        self.features = original_model.features
        del original_model
        self.flatten = nn.Flatten()
        if arch.startswith('vgg16'):
            self.classifier = nn.Sequential(OrderedDict([
                ('do1', nn.Dropout()),
                ('fc1', nn.Linear(25088, 4096)),
                ('fc_relu1', nn.ReLU(inplace=True)),
                ('do2', nn.Dropout()),
                ('fc2', nn.Linear(4096, 4096)),
                ('fc_relu2', nn.ReLU(inplace=True)),
                ('fc3', nn.Linear(4096, num_classes))
            ]))
            self.modelName = 'vgg16'
        elif arch.startswith('alexnet'):
            self.classifier = nn.Sequential(
                # nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                # nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes)
            )
            self.modelName = 'alexnet'
        else:
            raise ("Finetuning not supported on this architecture yet")

        # 特征层不学习
        # self.features_learning(False)

    def features_learning(self, flag=True):
        for param in self.features.parameters():
            param.requires_grad = flag

    def classifier_learning(self, flag=True):
        for param in self.classifier.parameters():
            param.requires_grad = flag

    def forward(self, x: torch.Tensor):
        in_size = x.size(0)
        x = self.features(x)
        # x = x.view(in_size, -1)  # flatten the tensor
        x = self.flatten(x)
        x = self.classifier(x)
        return x

# new vgg in cifar. input by 32*32. Original is 224*224 resized
class VGG16_CIFAR10(nn.Module):
    def __init__(self):
        super().__init__()
        num_classes = 10
        self.features = models.vgg16(pretrained=True).features
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(OrderedDict([
            ('do1', nn.Dropout()),
            ('fc1', nn.Linear(512, 4096)),
            ('fc_relu1', nn.ReLU(inplace=True)),
            ('do2', nn.Dropout()),
            ('fc2', nn.Linear(4096, 4096)),
            ('fc_relu2', nn.ReLU(inplace=True)),
            ('fc3', nn.Linear(4096, num_classes))
        ]))
        # self.features_learning(False)

    def features_learning(self, flag=True):
        for param in self.features.parameters():
            param.requires_grad = flag

    def classifier_learning(self, flag=True):
        for param in self.classifier.parameters():
            param.requires_grad = flag

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

class VGG16_CIFAR100(nn.Module):
    def __init__(self):
        super().__init__()
        num_classes = 100
        self.features = models.vgg16(pretrained=True).features
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(OrderedDict([
            ('do1', nn.Dropout()),
            ('fc1', nn.Linear(512, 4096)),
            ('fc_relu1', nn.ReLU(inplace=True)),
            ('do2', nn.Dropout()),
            ('fc2', nn.Linear(4096, 4096)),
            ('fc_relu2', nn.ReLU(inplace=True)),
            ('fc3', nn.Linear(4096, num_classes))
        ]))
        # self.features_learning(False)

    def features_learning(self, flag=True):
        for param in self.features.parameters():
            param.requires_grad = flag

    def classifier_learning(self, flag=True):
        for param in self.classifier.parameters():
            param.requires_grad = flag

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxp = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgp = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxp(x)  # add maxp
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgp(x)  # new avgp
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


if __name__ == '__main__':
    # Dummy arguments
    class Args:
        def __init__(self):
            self.arch = "vgg16"


    args = Args()

    # n = Net(None, args()) #Network on ImageNet (224 * 224 * 3)
    # print(n(torch.zeros(2, 3, 32, 32)).shape)

    VGG_Alex('alexnet')

    model = {
        'alexnet': VGG_Alex('alexnet'),
        'vgg16': VGG_Alex('vgg16'),
        'resnet18': ResNet18(),
        'resnet50': ResNet50(),
    }[args.arch.lower()]

    # n = Net() #vgg16 on cifar10 (32 * 32 * 3)
    n = Net('resnet18')  # resnet on cifar10 (32 * 32 * 3)
    # x = Variable(torch.FloatTensor(2, 3, 40, 40))
    print(n(torch.zeros(2, 3, 32, 32)).shape)
