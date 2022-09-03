# -*- encoding: utf-8 -*-
# Train, Prune, LRP True & False

from __future__ import print_function

import time

import numpy as np
import torch
import copy

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import models, datasets, transforms
from torch.autograd import Variable, Function
import modules.data as dataset
from modules.network import VGG_Alex
from modules.lrp import lrp_prune
from modules.prune_layer import *
from modules.flop import *
from modules.flops_counter import add_flops_counting_methods
from collections import OrderedDict
from operator import itemgetter
from heapq import nsmallest
import os


def save_in_out_data(self, input, output):
    self.input = input[0]
    self.output = output.data


class FilterPrunner:
    def __init__(self, model, args):
        self.model = model
        self.reset()
        self.args = args

    def reset(self):
        self.filter_ranks = {}
        self.forward_hook()

    def forward_hook(self):
        # For Forward Hook
        for name, module in self.model.features._modules.items():
            module.register_forward_hook(save_in_out_data)
        for name, module in self.model.classifier._modules.items():
            module.register_forward_hook(save_in_out_data)

    def forward_lrp(self, x):
        in_size = x.size(0)
        self.activation_to_layer = {}
        self.grad_index = 0

        self.activation_index = 0
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            x = module(x)
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                self.activation_to_layer[self.activation_index] = layer
                self.activation_index += 1

        x = x.view(in_size, -1)  # flatten the tensor
        # x = x.view(in_size, 256 * 6 * 6) #alexnet
        return self.model.classifier(x)

    def backward_lrp(self, R, relevance_method='z'):
        for name, module in enumerate(self.model.classifier[::-1]):  # 접근 방법
            # print(R[10,:].sum())
            R = lrp_prune(module, R.data, relevance_method, 1)

        for name, module in enumerate(self.model.features[::-1]):  # 접근 방법
            # print(R[10, :].sum())
            if isinstance(module, torch.nn.modules.conv.Conv2d):  # !!!
                activation_index = self.activation_index - self.grad_index - 1

                values = \
                    torch.sum(R, dim=0, keepdim=True). \
                        sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)[0, :, 0, 0].data

                if activation_index not in self.filter_ranks:
                    self.filter_ranks[activation_index] = torch.FloatTensor(
                        R.size(1)).zero_().cuda() if self.args.cuda else torch.FloatTensor(
                        R.size(1)).zero_()

                self.filter_ranks[activation_index] += values
                self.grad_index += 1

            R = lrp_prune(module, R.data, relevance_method, 1)

    def forward(self, x):
        self.activations = []  # 전체 conv_layer의 activation map 수
        self.weights = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}  # conv layer의 순서 7: 17의미는 7번째 conv layer가 전체에서 17번째에 있다라는 뜻

        activation_index = 0
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            x = module(x)  # 일반적인 forward를 수행하면서..
            if isinstance(module, torch.nn.modules.conv.Conv2d):  # conv layer 일때 여기를 지나감
                x.register_hook(self.compute_rank)
                if self.args.method_type == 'weight':
                    self.weights.append(module.weight)
                self.activations.append(x)
                self.activation_to_layer[activation_index] = layer
                activation_index += 1

        return self.model.classifier(x.view(x.size(0), -1))

    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1  # 뒤에서부터 하나씩 끄집어 냄
        activation = self.activations[activation_index]

        if self.args.method_type == 'ICLR':
            values = \
                torch.sum((activation * grad), dim=0, keepdim=True). \
                    sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)[0, :, 0, 0].data  # P. Molchanov et al., ICLR 2017
            # Normalize the rank by the filter dimensions
            values = \
                values / (activation.size(0) * activation.size(2) * activation.size(3))

        elif self.args.method_type == 'grad':
            values = \
                torch.sum((grad), dim=0, keepdim=True). \
                    sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)[0, :, 0, 0].data  # # X. Sun et al., ICML 2017
            # Normalize the rank by the filter dimensions
            values = \
                values / (activation.size(0) * activation.size(2) * activation.size(3))

        elif self.args.method_type == 'weight':
            weight = self.weights[activation_index]
            values = \
                torch.sum((weight).abs(), dim=1, keepdim=True). \
                    sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)[:, 0, 0,
                0].data  # Many publications based on weight and activation(=feature) map

        else:
            raise ValueError('No criteria')

        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = torch.FloatTensor(
                activation.size(1)).zero_().cuda() if self.args.cuda else torch.FloatTensor(activation.size(1)).zero_()

        self.filter_ranks[activation_index] += values
        self.grad_index += 1

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:

            if self.args.relevance:  # average over trials - LRP case (this is not normalization !!)
                v = self.filter_ranks[i]
                v = v / torch.sum(v)  # torch.sum(v) = total number of dataset
                self.filter_ranks[i] = v.cpu()
            else:
                if self.args.norm:  # L2-norm for global rescaling
                    if self.args.method_type == 'weight':  # weight & L1-norm (Li et al., ICLR 2017)
                        v = self.filter_ranks[i]
                        v = v / torch.sum(v)  # L1
                        # v = v / torch.sqrt(torch.sum(v * v)) #L2
                        self.filter_ranks[i] = v.cpu()
                    elif self.args.method_type == 'ICLR':  # |grad*act| & L2-norm (Molchanov et al., ICLR 2017)
                        v = torch.abs(self.filter_ranks[i])
                        v = v / torch.sqrt(torch.sum(v * v))
                        self.filter_ranks[i] = v.cpu()
                    elif self.args.method_type == 'grad':  # |grad| & L2-norm (Sun et al., ICML 2017)
                        v = torch.abs(self.filter_ranks[i])
                        v = v / torch.sqrt(torch.sum(v * v))
                        self.filter_ranks[i] = v.cpu()
                else:
                    if self.args.method_type == 'weight':  # weight
                        v = self.filter_ranks[i]
                        self.filter_ranks[i] = v.cpu()
                    elif self.args.method_type == 'ICLR':  # |grad*act|
                        v = torch.abs(self.filter_ranks[i])
                        self.filter_ranks[i] = v.cpu()
                    elif self.args.method_type == 'grad':  # |grad|
                        v = torch.abs(self.filter_ranks[i])
                        self.filter_ranks[i] = v.cpu()

    def get_prunning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)
        # filters_to_prune: filters to be pruned 1) layer number, 2) filter number, 3) its value

        # After each of the k filters are prunned,
        # the filter index of the next filters change since the model is smaller.
        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
            for i in range(len(filters_to_prune_per_layer[l])):  # 이거 꽤 중요함!
                # 앞 뒤 layer를 잘라줘야 하므로 자르기 전에 미리 차감 시켜놓아야됨
                filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))

        return filters_to_prune  # 잘라야 할 filter들의 1) layer number, 2) filter number

    def lowest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))
                # data 변수에 모든 layer의 모든 filter의 값을 쭈욱 나열 시킨다.

        return nsmallest(num, data, itemgetter(2))  # data list 내에서 가장 작은 수를 num(=512개) 만큼 뽑아서 리스트에 저장


class PruningFineTuner:
    def __init__(self, args, model: VGG_Alex):
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

        self.args = args
        self.setup_dataloaders()
        self.model = model

        self.criterion = nn.CrossEntropyLoss()
        self.prunner = FilterPrunner(self.model, args)
        self.model.train()  # set model parameters to trainable
        self.save_loss = False

    def setup_dataloaders(self):
        kwargs = {'num_workers': self.args.num_workers, 'pin_memory': self.args.pin_memory}

        # Data Acquisition
        get_dataset = {
            "cifar10": dataset.get_cifar10,  # CIFAR-10
            'cifar10_raw': dataset.get_cifar10_raw,
            'cifar100_raw': dataset.get_cifar100_raw,
            'imagenet': dataset.get_imagenet,  # ImageNet
        }[self.args.data_type.lower()]
        train_dataset, test_dataset = get_dataset()
        print(f"train_dataset:{len(train_dataset)}, test_dataset:{len(test_dataset)}")
        # Data Loader (Input Pipeline)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                        batch_size=self.args.train_batch_size,
                                                        shuffle=True,
                                                        **kwargs)

        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                       batch_size=self.args.test_batch_size,
                                                       shuffle=False, **kwargs)

        self.train_size = len(train_dataset)
        self.test_size = len(test_dataset)
        self.train_num = len(self.train_loader)
        self.test_num = len(self.test_loader)

    def test_only(self):
        with torch.no_grad():
            self.model.eval()
            correct = torch.zeros(1).cuda()
            loss = torch.zeros(1).cuda()
            t = time.time()
            for x, y in self.train_loader:
                if self.args.cuda:
                    x, y = x.cuda(), y.cuda()
                x, y = Variable(x), Variable(y)
                logits = self.model(x)
                loss += self.criterion(logits, y)
                pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                correct += torch.eq(pred, y).sum()
            loss /= self.train_num
            correct = correct.item()
            loss = loss.item()
            print(f'Train Loss: {loss:.4f}, Accuracy: {correct}/{self.train_size}'
                  f' ({100. * correct / self.train_size:.2f}%)')
            print(f'check train acc using {time.time() - t:.2f}s')

            loss = torch.zeros(1).cuda()
            correct = torch.zeros(1).cuda()
            for x, y in self.test_loader:
                if self.args.cuda:
                    x, y = x.cuda(), y.cuda()
                x, y = Variable(x), Variable(y)
                logits = self.model(x)

                loss += self.criterion(logits, y).item()
                # get the index of the max log-probability
                pred = logits.data.max(1, keepdim=True)[1]
                correct += pred.eq(y.data.view_as(pred)).cpu().sum()
            loss /= self.test_num
            correct = correct.item()
            loss = loss.item()
            print(f'Test loss: {loss:.4f}, Accuracy: {correct}/{self.test_size} '
                  f'({100. * correct / self.test_size:.2f}%)')
            self.model.train()

    def train_only(self, epoches=10):
        if 1:
            optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
            print(f'using SGD..')
        else:
            optimizer = optim.Adam(self.model.classifier.parameters(), lr=self.args.lr)
            print('using adam')
        shed1 = optim.lr_scheduler.StepLR(optimizer, 10, 0.1)
        # self.test_only()
        for i in range(1, epoches + 1):
            # if i == 1:
            #     self.model.features_learning(False)
            # elif i == 6:
            #     self.model.features_learning(True)
            print('-' * 20 + '\n' + "Epoch: ", i)
            t = time.time()
            for batch_idx, (x, y) in enumerate(self.train_loader):
                if self.args.cuda:
                    x, y = x.cuda(), y.cuda()
                x, y = Variable(x), Variable(y)
                optimizer.zero_grad()
                loss = self.criterion(self.model(x), y)
                loss.backward()
                optimizer.step()
            if 1:
                print(shed1.get_last_lr())
                shed1.step()
            print(f'training one epoch using {time.time() - t:.2f}s')
            self.test_only()

    def prune_of_sample(self, dataset):
        self.train_loss_tot = []
        self.test_loss_tot = []
        self.test_acc_tot = []
        self.test_iter = []
        self.flop_val = []
        self.num_param = []
        self.R_tot = []
        self.data_tot = []
        self.time_tot = []
        self.save_loss = True

        # Get the accuracy before prunning
        self.niter = 0
        self.temp = 0
        self.test()
        self.model.train()

        # Make sure all the layers are trainable
        # for param in self.model.features.parameters():
        #     param.requires_grad = True
        self.model.features_learning(True)

        number_of_filters = self.total_num_filters()
        num_filters_to_prune_per_iteration = int(number_of_filters * self.args.pr_step)  # 0.05 (5%) -> 0.01 (1%) temporally
        iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)
        iterations = int(iterations * self.args.total_pr)  # up to 80%

        for kk in range(1, iterations + 1):
            print('-' * 20, '\n', "Ranking filters.. step {}".format(kk))
            self.niter += 1

            self.prunner.reset()
            self.run_batch()
            self.prunner.normalize_ranks_per_layer()
            prune_targets = self.prunner.get_prunning_plan(num_filters_to_prune_per_iteration)

            # prune_targets: 잘라야 할 filter들의 1) layer number, 2) filter number가 넘어옴
            layers_prunned = {}
            for layer_index, filter_index in prune_targets:
                if layer_index not in layers_prunned:
                    layers_prunned[layer_index] = 0
                layers_prunned[layer_index] += 1

            print("Layers that will be prunned", layers_prunned)  # 총 잘릴 layer 별 filter 수
            print("Prunning filters.. ")
            model = self.model.cpu()  # 현재 모델 갖다가..
            for layer_index, filter_index in prune_targets:  # 하나씩 꺼내서 자르기 시작
                model = prune_conv_layer_sequential(model, layer_index, filter_index, cuda_flag=self.args.cuda)

            self.model = model.cuda() if self.args.cuda else model

            message = str(100 * float(self.total_num_filters()) / number_of_filters) + "%"
            print("Filters prunned", str(message))
            self.test()  # 잘리고 나서 test 해봄
            if self.args.fine_tune:
                print("Fine tuning to recover from prunning iteration.")
                optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
                self.train(optimizer, epoches=self.args.epochs)



    def run_batch(self, x_batch, y_batch):
        self.model.zero_grad()
        if self.args.relevance:  # lrp_based
            output = self.prunner.forward_lrp(x_batch)

            T = torch.zeros_like(output)
            for ii in range(len(y_batch)):
                T[ii, y_batch[ii]] = 1.0

            self.prunner.backward_lrp(T.data)

            # print(f'Train Epoch: [{batch_idx * len(batch)}/{len(self.train_loader.dataset)} ({100. * batch_idx / len(self.train_loader):.0f}%)]')
            loss = self.criterion(output, y_batch)
            self.train_loss_batch += loss.item()

        else:  # gradient_based
            output = self.prunner.forward(x_batch)
            loss = self.criterion(output, y_batch)
            loss.backward()
            # print(f'Train Epoch: [{batch_idx * len(batch)}/{len(self.train_loader.dataset)} ({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            self.train_loss_batch += loss.item()


    def train(self, optimizer=None, epoches=10):
        if optimizer is None:
            optimizer = optim.SGD(self.model.classifier.parameters(), lr=self.args.lr, momentum=self.args.momentum)
            # optimizer = optim.Adam(self.model.classifier.parameters(), lr=self.args.lr)
        print(f'using SGD..')

        # self.test()
        # self.correct = 0
        for i in range(1, epoches + 1):
            print("Epoch: ", i)
            self.train_epoch(optimizer)
            self.test()
        print("Finished fine tuning")
        # self.correct /= epoches

    def train_epoch(self, optimizer=None, rank_filters=False):
        t = time.time()
        self.train_loss_batch = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            # data, target = Variable(data), Variable(target)
            self.train_batch(optimizer, batch_idx, data, target, rank_filters)
        print(f'training one epoch using {time.time() - t:.2f}s')
        if self.save_loss:
            self.train_loss_tot.append(self.train_loss_batch / len(self.train_loader.dataset))

    def train_batch(self, optimizer, batch_idx, batch, label, rank_filters):
        self.model.zero_grad()

        if rank_filters:
            if self.args.relevance:  # lrp_based
                output = self.prunner.forward_lrp(batch)

                T = torch.zeros_like(output)
                for ii in range(len(label)):
                    T[ii, label[ii]] = 1.0

                self.prunner.backward_lrp(T.data)

                # print(f'Train Epoch: [{batch_idx * len(batch)}/{len(self.train_loader.dataset)} ({100. * batch_idx / len(self.train_loader):.0f}%)]')
                loss = self.criterion(output, label)
                self.train_loss_batch += loss.item()

            else:  # gradient_based
                output = self.prunner.forward(batch)
                loss = self.criterion(output, label)
                loss.backward()
                # print(f'Train Epoch: [{batch_idx * len(batch)}/{len(self.train_loader.dataset)} ({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
                self.train_loss_batch += loss.item()
        else:
            loss = self.criterion(self.model(batch), label)
            loss.backward()
            optimizer.step()
            # print(f'Train Epoch: [{batch_idx * len(batch)}/{len(self.train_loader.dataset)} ({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            self.train_loss_batch += loss.item()



    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        for x, y in self.test_loader:
            if self.args.cuda:
                x, y = x.cuda(), y.cuda()
            x, y = Variable(x), Variable(y)
            logits = self.model(x)

            test_loss += self.criterion(logits, y).item()
            # get the index of the max log-probability
            pred = logits.data.max(1, keepdim=True)[1]
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()
            del x, y, logits, pred

        test_loss /= self.test_size
        print(
            f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{self.test_size} ({100. * correct / self.test_size:.0f}%)\n')
        # self.correct += correct

        if self.save_loss:
            self.test_acc_tot.append((100. * correct).numpy() / self.test_size)
            self.test_loss_tot.append(test_loss)
            self.test_iter.append(self.niter)

            # # FLOP calculation
            # if True:
            #     sample_batch = torch.FloatTensor(1, 3, 32, 32).cuda()
            # else:
            #     sample_batch = torch.FloatTensor(1, 3, 224, 224).cuda()
            # self.model = add_flops_counting_methods(self.model)
            # self.model.eval().start_flops_count()
            # # _ = self.model(sample_batch)
            # self.flop_val.append(flops_to_string(self.model.compute_average_flops_cost()))
            # self.num_param.append(get_model_parameters_number(self.model))
            # print('Flops:  {}'.format(flops_to_string(self.model.compute_average_flops_cost())))
            # print('Params: ' + get_model_parameters_number(self.model))

        self.model.train()

    def total_num_filters(self):
        # Conv layer의 모든 filter 수를 counting
        filters = 0
        for name, module in self.model.features._modules.items():
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                filters = filters + module.out_channels
        return filters

    def get_candidates_to_prune(self, num_filters_to_prune):
        self.prunner.reset()

        self.train_epoch(rank_filters=True)
        # training 하면서 동시에 hook 써서 후보 찾기 #
        # (각 layer 마다 compute_rank 안에서 계산되어서 self.filter_ranks list에 저장이 된다.

        self.prunner.normalize_ranks_per_layer()  # Normalization

        return self.prunner.get_prunning_plan(num_filters_to_prune)

    def forward_hook(self):
        # For Forward Hook
        for name, module in self.model.features._modules.items():
            module.register_forward_hook(save_in_out_data)
        for name, module in self.model.classifier._modules.items():
            module.register_forward_hook(save_in_out_data)

    def prune(self):
        self.train_loss_tot = []
        self.test_loss_tot = []
        self.test_acc_tot = []
        self.test_iter = []
        self.flop_val = []
        self.num_param = []
        self.R_tot = []
        self.data_tot = []
        self.time_tot = []
        self.save_loss = True

        # Get the accuracy before prunning
        self.niter = 0
        self.temp = 0
        self.test()
        self.model.train()

        # Make sure all the layers are trainable
        # for param in self.model.features.parameters():
        #     param.requires_grad = True
        self.model.features_learning(True)

        number_of_filters = self.total_num_filters()
        num_filters_to_prune_per_iteration = int(
            number_of_filters * self.args.pr_step)  # 0.05 (5%) -> 0.01 (1%) temporally
        iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)
        iterations = int(iterations * self.args.total_pr)  # up to 80%

        # # print("Number of prunning iterations to reduce 67% filters", iterations)

        # R_tot, data_tot, time_tot = self.lrp()  # lrp using conventional model
        # self.R_tot.append(R_tot)

        for kk in range(1, iterations + 1):
            print('-' * 20, '\n', "Ranking filters.. step {}".format(kk))
            self.niter += 1
            prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iteration)
            # prune_targets: 잘라야 할 filter들의 1) layer number, 2) filter number가 넘어옴
            layers_prunned = {}
            for layer_index, filter_index in prune_targets:
                if layer_index not in layers_prunned:
                    layers_prunned[layer_index] = 0
                layers_prunned[layer_index] += 1

            print("Layers that will be prunned", layers_prunned)  # 총 잘릴 layer 별 filter 수
            print("Prunning filters.. ")
            model = self.model.cpu()  # 현재 모델 갖다가..
            for layer_index, filter_index in prune_targets:  # 하나씩 꺼내서 자르기 시작
                model = prune_conv_layer_sequential(model, layer_index, filter_index, cuda_flag=self.args.cuda)

            self.model = model.cuda() if self.args.cuda else model

            message = str(100 * float(self.total_num_filters()) / number_of_filters) + "%"
            print("Filters prunned", str(message))
            self.test()  # 잘리고 나서 test 해봄
            if self.args.fine_tune:
                print("Fine tuning to recover from prunning iteration.")
                optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
                self.train(optimizer, epoches=self.args.epochs)
            # R_tot, data_tot, time_tot = self.lrp()
            # self.R_tot.append(R_tot)
            # del R_tot

        # print("Finished. Going to fine tune the model a bit more")
        # self.niter += 1
        # self.train(optimizer, epoches=5)
