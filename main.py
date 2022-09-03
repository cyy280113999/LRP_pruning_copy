# -*- encoding: utf-8 -*-
'''
@reference: Seul-Ki Yeom et al., "Pruning by explaining: a novel criterion for deep neural network pruning," Pattern Recognition, 2020.
@author: Seul-Ki Yeom, Philipp Seegerer, Sebastian Lapuschkin, Alexander Binder, Simon Wiedemann, Klaus-Robert Müller, Wojciech Samek
'''

from __future__ import print_function
import argparse
import sys

import numpy as np
import torch
import os

from modules.network import ResNet18, ResNet50, VGG_Alex, VGG16_CIFAR10, VGG16_CIFAR100
import modules.prune_resnet as modules_resnet
import modules.prune_vgg as modules_vgg


# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# base
# --arch alexnet --data-type cifar10 --train --prune --lr 0.01 --train-batch-size 512 --test-batch-size 512
# 74
#  --arch alexnet --data-type cifar10 --train --prune --epoch 3 -lr 0.001 -m 0.9 -trsz 64 -tesz 64
# 78
# --arch alexnet --data-type cifar10 --method-type lrp --train --prune --epoch 10 -lr 0.0005 -m 0.9 -trsz 64 -tesz 64
# success pruning
# --arch alexnet --data-type cifar10 --method-type lrp --resume --prune --fine-tune -lr 0.001 -trsz 64 -tesz 64

# --arch vgg16 --data-type cifar10 --train --prune --lr 0.01 --train-batch-size 128 --test-batch-size 128

def get_args():
    # params compatibility:
    # args and config file required same params
    # args and config file required same type
    # behavior when loss params ...

    check_true = lambda x:x in ['True','true']
    parser = argparse.ArgumentParser(description='PyTorch VGG16 based ImageNet')
    parser.add_argument('--arch', '-ar', type=str, default='vgg16', metavar='ARCH',
                        help='model architecture: resnet18, resnet50, vgg16, alexnet')
    parser.add_argument('--data-type', '-ds', type=str, default='cifar10', metavar='N',
                        help='model architecture selection: cifar10/imagenet')
    parser.add_argument('--method-type', '-cr', type=str, default='lrp', metavar='N',
                        help='model architecture selection: grad/taylor/weight/lrp')

    # trainning and restore
    parser.add_argument('--train', type=check_true, default='false', help='training with data')
    parser.add_argument('--save', type=check_true, default='false',help='save model')
    parser.add_argument('--resume', type=check_true, default='false',
                        help='if we have pretrained model')
    parser.add_argument('--optimize_method','-optm', type=str, default='sgd')
    parser.add_argument('--epochs','-ep', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', '-lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: ?)')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: ?)')

    parser.add_argument('--prune', type=check_true, default='false',
                        help='pruning model')
    parser.add_argument('--prune_cnn', type=check_true, default='false',)
    parser.add_argument('--prune_fc', type=check_true, default='false',)
    parser.add_argument('--prune_train', type=check_true, default='false',)
    # wait, classes support for list
    cl_fun = lambda x:eval(x)
    parser.add_argument('--classes', type=cl_fun, default=None)
    parser.add_argument('--samples', type=cl_fun, default=None)

    parser.add_argument('--fine_tune', type=check_true, default='false',help='training after pruning')
    parser.add_argument('--fine_tune_conv', type=check_true, default='false')
    parser.add_argument('--fine_tune_fc', type=check_true, default='false')


    parser.add_argument('--train_batch_size', '-trsz', type=int, default=256, metavar='N',
                        help='input batch size for training (default: ?)')
    parser.add_argument('--test_batch_size', '-tesz', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: ?)')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--pin_memory', type=check_true, default='true')
    parser.add_argument('--persistent_workers', type=check_true, default='true')


    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no_cuda', type=check_true, default='false',
                        help='disables CUDA training')
    parser.add_argument('--norm', type=check_true, default='true', help='add normalization')
    parser.add_argument('--total_pr', type=float, default=9.001 / 10.0, metavar='M',
                        help='Total pruning rate')
    parser.add_argument('--pr_step', type=float, default=0.05, metavar='M',
                        help='Pruning step: 0.05 (5% for each step)')
    parser.add_argument('--showlrp', type=check_true, default='false')
    parser.add_argument('--log_name', type=str, default='log/log.txt', help='saved name')

    # other args translating
    # redundant, wait for remove
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # lrp pruning criterion
    args.relevance = True if args.method_type == 'lrp' else False

    return args


class Config:
    def __init__(self, dict_):
        self.__dict__.update(dict_)
        self.cuda = not self.no_cuda and torch.cuda.is_available()
        self.relevance = True if self.method_type == 'lrp' else False


def load_config(filename):
    import json
    with open(filename) as json_file:
        config = Config(json.load(json_file))
    return config


if __name__ == '__main__':
    import torch.backends.cudnn

    torch.backends.cudnn.benchmark = True
    if '--config_file' == sys.argv[1]:
        config = load_config('config.json')
        args = config
    else:
        args = get_args()

    if args.arch == 'alexnet':
        model = VGG_Alex(arch='alexnet')
    elif args.arch == 'vgg16':
        model = VGG_Alex(arch='vgg16')
    elif args.arch == 'resnet18':
        model = ResNet18()
    elif args.arch == 'resnet50':
        model = ResNet50()
    elif args.arch == 'vgg16_cifar10':
        model = VGG16_CIFAR10()
    elif args.arch == 'vgg16_cifar100':
        model = VGG16_CIFAR100()
    else:
        raise ValueError('unvalid model name')

    if args.cuda:
        model = model.cuda()

    if args.resume:
        save_loc = f"./checkpoint/{args.arch}_{args.data_type}_ckpt.pth"
        model.load_state_dict(torch.load(save_loc))

    if args.arch.lower() in ['resnet18', 'resnet50']:
        fine_tuner = modules_resnet.PruningFineTuner(args, model)
    elif args.arch.lower() in ['vgg16', 'alexnet', 'vgg16_cifar10', 'vgg16_cifar100']:
        fine_tuner = modules_vgg.PruningFineTuner(args, model)
    else:
        raise Exception(f'not valid model :{args.arch}')


    if args.showlrp:
        fine_tuner.showlrp()
        print(' ')

    if args.train:
        print(f'Start training! Dataset: {args.data_type}, Architecture: {args.arch}, Epoch: {args.epochs}')
        fine_tuner.train_only(epoches=args.epochs,eval_train=True,eval_test=True)
        if args.save or input('Save Model ?') == 'y':
            file_path = f"./checkpoint/{args.arch}_{args.data_type}_ckpt.pth"
            if not os.path.exists(os.path.dirname(file_path)):
                os.mkdir(os.path.dirname(file_path))
            torch.save(model.state_dict(), file_path)

    if args.prune:
        print(
            f'Start pruning! Dataset: {args.data_type}, Architecture: {args.arch}, Pruning Method: {args.method_type},'
            f' Total Pruning rate: {args.total_pr}, Pruning step: {args.pr_step}')
        fine_tuner.prune()

