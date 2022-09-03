import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.lrp_general6 import *

class Cannotloadmodelweightserror(Exception):
    pass

class Modulenotfounderror(Exception):
    pass

class BasicBlock_kuangliu_c(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_kuangliu_c, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        self.elt = sum_stacked2()
        self.somerelu= nn.ReLU()

    def forward(self, x):
        out = self.somerelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        #out += self.shortcut(x)
        out = self.elt( torch.stack([out, self.shortcut(x) ], dim=0))
        out = self.somerelu(out)
        return out


class Bottleneck_kuangliu_c(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck_kuangliu_c, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        self.elt = sum_stacked2()
        self.somerelu= nn.ReLU()

    def forward(self, x):
        out = self.somerelu(self.bn1(self.conv1(x)))
        out = self.somerelu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        #out += self.shortcut(x)
        out = self.elt( torch.stack([out, self.shortcut(x) ], dim=0))
        out = self.somerelu(out)
        return out


class ResNet_kuangliu_c(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_kuangliu_c, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.somerelu= nn.ReLU()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out =  self.somerelu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        #out = F.avg_pool2d(out, 4)
        #out = out.view(out.size(0), -1)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        out = self.linear(out)
        return out

    def setbyname(self, name, value):

        def iteratset(obj, components, value):

            if not hasattr(obj, components[0]):
                return False
            elif len(components) == 1:
                setattr(obj, components[0], value)
                # print('found!!', components[0])
                # exit()
                return True
            else:
                nextobj = getattr(obj, components[0])
                return iteratset(nextobj, components[1:], value)

        components = name.split('.')
        success = iteratset(self, components, value)
        return success


    def copyfromresnet(self, net, lrp_params, lrp_layer2method):
        # assert not (isinstance(net, ResNet) or isinstance(net, ResNet_kuangliu_c))
        assert not isinstance(net, ResNet_kuangliu_c)

        # --copy linear
        # --copy conv2, while fusing bns
        # --reset bn

        # first conv, then bn,
        # means: when encounter bn, find the conv before -- implementation dependent

        updated_layers_names = []

        last_src_module_name = None
        last_src_module = None

        for src_module_name, src_module in net.named_modules():
            #print('at src_module_name', src_module_name)

            foundsth = False

            if isinstance(src_module, nn.Linear):
                # copy linear layers
                foundsth = True
                #print('is Linear')
                # m =  oneparam_wrapper_class( copy.deepcopy(src_module) , linearlayer_eps_wrapper_fct(), parameter1 = linear_eps )
                wrapped = get_lrpwrapperformodule(copy.deepcopy(src_module),
                                                  lrp_params, lrp_layer2method)
                if False == self.setbyname(src_module_name, wrapped):
                    raise Modulenotfounderror(
                        "could not find module " + src_module_name + " in target net to copy")
                updated_layers_names.append(src_module_name)
            # end of if

            if isinstance(src_module, nn.Conv2d):
                # store conv2d layers
                foundsth = True
                #print('is Conv2d')
                last_src_module_name = src_module_name
                last_src_module = src_module
            # end of if

            if isinstance(src_module, nn.BatchNorm2d):
                # conv-bn chain
                foundsth = True
                #print('is BatchNorm2d')

                if (True == lrp_params['use_zbeta']) and (
                        last_src_module_name == 'conv1'):
                    thisis_inputconv_andiwant_zbeta = True
                else:
                    thisis_inputconv_andiwant_zbeta = False

                m = copy.deepcopy(last_src_module)
                m = bnafterconv_overwrite_intoconv(m, bn=src_module) # outcomment if you want no conv-bn fusion
                # wrap conv
                wrapped = get_lrpwrapperformodule(m, lrp_params,
                                                  lrp_layer2method,
                                                  thisis_inputconv_andiwant_zbeta=thisis_inputconv_andiwant_zbeta)

                if False == self.setbyname(last_src_module_name, wrapped):
                    raise Modulenotfounderror(
                        "could not find module " + nametofind + " in target net to copy")

                updated_layers_names.append(last_src_module_name)

                # wrap batchnorm
                wrapped = get_lrpwrapperformodule(resetbn(src_module), lrp_params, lrp_layer2method) # outcomment if you want no conv-bn fusion
                #wrapped = get_lrpwrapperformodule(src_module, lrp_params, lrp_layer2method) # outcomment if you want no conv-bn fusion
                if False == self.setbyname(src_module_name, wrapped):
                    raise Modulenotfounderror(
                        "could not find module " + src_module_name + " in target net to copy")
                updated_layers_names.append(src_module_name)
            # end of if

            # if False== foundsth:
            #  print('!untreated layer')
            #   print('\n')

        # sum_stacked2 is present only in the targetclass, so must iterate here
        for target_module_name, target_module in self.named_modules():

            if isinstance(target_module,
                          (nn.ReLU, nn.AdaptiveAvgPool2d, nn.MaxPool2d)):
                wrapped = get_lrpwrapperformodule(target_module, lrp_params,
                                                  lrp_layer2method)

                if False == self.setbyname(target_module_name, wrapped):
                    raise Modulenotfounderror(
                        "could not find module " + src_module_name + " in target net to copy")
                updated_layers_names.append(target_module_name)

            if isinstance(target_module, sum_stacked2):

                wrapped = get_lrpwrapperformodule(target_module, lrp_params,
                                                  lrp_layer2method)
                if False == self.setbyname(target_module_name, wrapped):
                    raise Modulenotfounderror(
                        "could not find module " + target_module_name + " in target net , impossible!")
                updated_layers_names.append(target_module_name)
        '''
        for target_module_name, target_module in self.named_modules():
            if target_module_name not in updated_layers_names:
                print('not updated:', target_module_name)
        '''


def ResNet18_kuangliu_c():
    return ResNet_kuangliu_c(BasicBlock_kuangliu_c, [2, 2, 2, 2])

def ResNet50_kuangliu_c():
    return ResNet_kuangliu_c(Bottleneck_kuangliu_c, [3, 4, 6, 3])
