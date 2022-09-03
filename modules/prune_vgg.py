# -*- encoding: utf-8 -*-
# Train, Prune, LRP True & False

from __future__ import print_function

import time
import tqdm
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
import matplotlib.pyplot as plt


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
        self.prunedLayerIdx2ModuleIdx = {}
        self.grad_index = 0

        self.layer_counter = -1
        for module_idx, (name, module) in enumerate(self.model.features._modules.items()):
            x = module(x)
            if isinstance(module, torch.nn.modules.conv.Conv2d) and self.args.prune_cnn:
                self.layer_counter+=1
                self.prunedLayerIdx2ModuleIdx[self.layer_counter] = module_idx

        x = x.view(x.size(0), -1)

        for module_idx, (name, module) in enumerate(self.model.classifier._modules.items(),start=module_idx+1):
            x = module(x)
            if isinstance(module, torch.nn.Linear) and self.args.prune_fc:
                self.layer_counter += 1
                self.prunedLayerIdx2ModuleIdx[self.layer_counter] = module_idx
        # last fc not prune ! in backward
        return x

    def backward_lrp(self, R, relevance_method='z+'):
        self.layer_counter+=1
        last_fc_flag=False
        for name, module in enumerate(self.model.classifier[::-1]):  # 접근 방법
            if isinstance(module, torch.nn.Linear) and self.args.prune_fc:  # !!!
                self.layer_counter-=1
                # activation_index = self.layer_counter - self.grad_index
                if not last_fc_flag:
                    last_fc_flag=True
                    self.prunedLayerIdx2ModuleIdx.pop(self.layer_counter)
                else:
                    if self.layer_counter not in self.filter_ranks:
                        self.filter_ranks[self.layer_counter] = torch.zeros_like(R[0])

                    self.filter_ranks[self.layer_counter] += torch.sum(R, dim=0)
                    # self.grad_index += 1
            R = lrp_prune(module, R.data, relevance_method, 1)

        # lrp of conv include reshaping

        for name, module in enumerate(self.model.features[::-1]):  # 접근 방법
            if isinstance(module, torch.nn.modules.conv.Conv2d) and self.args.prune_cnn:
                self.layer_counter -= 1
                # activation_index = self.layer_counter - self.grad_index

                values = torch.sum(R, dim=0, keepdim=True).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)[0, :, 0, 0].data

                if self.layer_counter not in self.filter_ranks:
                    self.filter_ranks[self.layer_counter] = torch.FloatTensor(
                        R.size(1)).zero_().cuda() if self.args.cuda else torch.FloatTensor(
                        R.size(1)).zero_()

                self.filter_ranks[self.layer_counter] += values
                # self.grad_index += 1

            R = lrp_prune(module, R.data, relevance_method, 1)
        return R

    def forward(self, x):
        self.activations = []  # 전체 conv_layer의 activation map 수
        self.weights = []
        self.gradients = []
        self.grad_index = 0
        self.prunedLayerIdx2ModuleIdx = {}  # conv layer의 순서 7: 17의미는 7번째 conv layer가 전체에서 17번째에 있다라는 뜻

        activation_index = 0
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            x = module(x)  # 일반적인 forward를 수행하면서..
            if isinstance(module, torch.nn.modules.conv.Conv2d):  # conv layer 일때 여기를 지나감
                x.register_hook(self.compute_rank)
                if self.args.method_type == 'weight':
                    self.weights.append(module.weight)
                self.activations.append(x)
                self.prunedLayerIdx2ModuleIdx[activation_index] = layer
                activation_index += 1

        return self.model.classifier(x.view(x.size(0), -1))

    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1  # 뒤에서부터 하나씩 끄집어 냄
        activation = self.activations[activation_index]

        if self.args.method_type == 'ICLR': # why name is ICLR ? it seems like taylor
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

    def get_pruned_filters_parallel(self, num_filters_to_prune):
        # remove score of data[], add separate store of scores
        # remove output scores
        # remove reading score statement
        data = []
        only_scores = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                # data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))
                data.append((self.prunedLayerIdx2ModuleIdx[i], j))
                only_scores.append(self.filter_ranks[i][j])
                # data 변수에 모든 layer의 모든 filter의 값을 쭈욱 나열 시킨다.

        sort_idx = torch.argsort(torch.tensor(only_scores).cpu())
        select_idx = sort_idx[:num_filters_to_prune].numpy()
        filters_to_prune = np.asarray(data, dtype=np.int)[select_idx]

        filters_to_prune_per_layer = {}
        for (l, f) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        return filters_to_prune_per_layer


class PruningFineTuner:
    def __init__(self, args, model: VGG_Alex):
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            np.random.seed(args.seed)

        self.args = args
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.setup_dataloaders()

        self.prunner = FilterPrunner(self.model, args)
        self.model.train()  # set model parameters to trainable
        self.save_loss = False
    def setup_dataloaders(self):
        kwargs = {'num_workers': self.args.num_workers, 'pin_memory': self.args.pin_memory,
                  'persistent_workers':self.args.persistent_workers}

        # Data Acquisition
        get_dataset = {
            "cifar10": dataset.get_cifar10,  # CIFAR-10
            'imagenet': dataset.get_imagenet,  # ImageNet
            'cifar10_raw': dataset.get_cifar10_raw,
            'cifar100_raw': dataset.get_cifar100_raw,
            'catsdogs':dataset.get_catsdogs,
        }[self.args.data_type.lower()]
        if self.args.data_type in ['cifar10','imagenet']:
            train_dataset, test_dataset = get_dataset()
            prune_dataset = train_dataset
        elif self.args.data_type in ['cifar10_raw','cifar100_raw'] and self.args.classes:  # choose class , sample . only for raw ds
            if self.args.data_type == 'cifar10_raw':
                cls_count = 10
            elif self.args.data_type == 'cifar100_raw':
                cls_count = 100
            else:
                raise Exception('only support cifar raw dataset.')
            if isinstance(self.args.classes,int):
                self.class_idxs = np.random.choice(cls_count, self.args.classes, replace=False)
                self.class_idxs.sort()
                self.class_idxs = self.class_idxs.tolist()
            elif isinstance(self.args.classes,list):
                self.class_idxs = self.args.classes
            else:
                raise TypeError(f'type out of [int/list],your:{type(self.args.classes)}')
            tqdm.tqdm.write(f'choose class {self.class_idxs}')
            self.map_to_old_class = {x: y for x, y in zip(range(len(self.class_idxs)), self.class_idxs)}
            self.map_to_new_class = {y: x for x, y in self.map_to_old_class.items()}

            train_dataset, prune_dataset, test_dataset = get_dataset(class_idxs=self.class_idxs, samples=self.args.samples)

            self.model = prune_to_classes(self.model, class_idxs=self.class_idxs, cuda_flag=self.args.cuda)
        elif self.args.data_type in['catsdogs']:
            train_dataset, prune_dataset, test_dataset = get_dataset(samples=self.args.samples,
                                                                     prune_from_train=self.args.prune_train)
        else:
            train_dataset, prune_dataset, test_dataset = get_dataset()


        tqdm.tqdm.write(f"train_dataset:{len(train_dataset)}, prune_dataset:{len(prune_dataset)}, test_dataset:{len(test_dataset)}")
        # Data Loader (Input Pipeline)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                        batch_size=self.args.train_batch_size,
                                                        shuffle=True,
                                                        **kwargs)

        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                       batch_size=self.args.test_batch_size,
                                                       shuffle=False, **kwargs)

        if not self.args.classes and not self.args.samples:
            self.prune_loader = self.train_loader
        else:
            self.prune_loader = torch.utils.data.DataLoader(dataset=prune_dataset,
                                                        batch_size=self.args.test_batch_size,
                                                        shuffle=False,
                                                        **kwargs)

        self.train_size = len(train_dataset)
        self.test_size = len(test_dataset)
        self.prune_size = len(prune_dataset)
        self.train_num = len(self.train_loader)
        self.test_num = len(self.test_loader)
        self.prune_iter_num = len(self.prune_loader)

        # test for new dataset and new model to check running correctly
        # if self.args.classes:
        #     print('test new model...')
        #     self.test_only()
        #     print('test finished.')

    def test_only(self, test_train=False):
        with torch.no_grad():
            self.model.eval()
            if test_train:
                correct = torch.zeros(1).cuda()
                loss = torch.zeros(1).cuda()
                # t = time.time()
                for x, y in tqdm.tqdm(self.train_loader,desc='testing train_ds'):
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
                acc=100. * correct / self.train_size
                tqdm.tqdm.write(f'Train Loss: {loss:.4f}, Accuracy: {correct}/{self.train_size}'
                      f' ({acc:.2f}%)')
                # print(f'check train acc using {time.time() - t:.2f}s')
            else:
                loss = torch.zeros(1).cuda()
                correct = torch.zeros(1).cuda()
                for x, y in tqdm.tqdm(self.test_loader,desc='testing test_ds'):
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
                acc=100. * correct / self.test_size
                tqdm.tqdm.write(f'Test loss: {loss:.4f}, Accuracy: {correct}/{self.test_size} '
                      f'({acc:.2f}%)')
        return acc

    def train_only(self, epoches=10,eval_train=False,eval_test=False):
        self.model.train()
        if hasattr(self.args,'optimize_method'):
            if self.args.optimize_method=='sgd':
                optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
                tqdm.tqdm.write(f'using SGD..')
            else:
                optimizer = optim.Adam(self.model.classifier.parameters(), lr=self.args.lr)
                tqdm.tqdm.write('using adam')
        else:
            optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
            tqdm.tqdm.write(f'using SGD..')
            optimizer = optim.Adam(self.model.classifier.parameters(), lr=self.args.lr)
            tqdm.tqdm.write('using adam')
        shed1 = optim.lr_scheduler.StepLR(optimizer, 20, 0.1)
        if eval_test:
            self.test_only(test_train=False)
        flood_threshold=0.01
        for i in range(1, epoches + 1):
            # if i == 1:
            #     self.model.features_learning(False)
            # elif i == 6:
            #     self.model.features_learning(True)
            tqdm.tqdm.write('-' * 10 + f"\nEpoch: {i}")
            t = time.time()
            for batch_idx, (x, y) in tqdm.tqdm(enumerate(self.train_loader), desc='train'):
                if self.args.cuda:
                    x, y = x.cuda(), y.cuda()
                x, y = Variable(x), Variable(y)
                optimizer.zero_grad()
                loss = self.criterion(self.model(x), y)
                if 1:
                    flood=(loss-flood_threshold).abs()+flood_threshold
                    loss=flood
                loss.backward()
                optimizer.step()
            if 1:
                tqdm.tqdm.write(str(shed1.get_last_lr()))
                shed1.step()
            tqdm.tqdm.write(f'training one epoch using {time.time() - t:.2f}s')
            if eval_train:
                self.test_only(test_train=True)
            if eval_test:
                self.test_only(test_train=False)
        # return acc



    def calculate_pruning_score(self):
        # t = time.time()
        self.train_loss_batch = 0
        for x_batch, y_batch in tqdm.tqdm(self.prune_loader,desc='calculate score'):
            if self.args.cuda:
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
            x_batch, y_batch = Variable(x_batch), Variable(y_batch)
            self.model.zero_grad()
            if self.args.relevance:  # lrp_based
                output = self.prunner.forward_lrp(x_batch)

                T = torch.zeros_like(output)
                for ii in range(len(y_batch)):
                    T[ii, y_batch[ii]] = 1.0
                # R = torch.nn.functional.one_hot(y_batch, 100)
                self.prunner.backward_lrp(T.data)

                loss = self.criterion(output, y_batch)
                self.train_loss_batch += loss.item()

            else:  # gradient_based
                output = self.prunner.forward(x_batch)
                loss = self.criterion(output, y_batch)
                loss.backward()
                self.train_loss_batch += loss.item()
        # print(f'calculate pruning score using {time.time() - t:.2f}s')
        if self.save_loss:
            self.train_loss_tot.append(self.train_loss_batch / self.prune_size)


    def total_num_filters(self):
        # Conv layer의 모든 filter 수를 counting
        filters = 0
        if self.args.prune_cnn:
            for name, module in self.model.features._modules.items():
                if isinstance(module, torch.nn.modules.conv.Conv2d):
                    filters = filters + module.out_channels
        if self.args.prune_fc:
            for name, module in self.model.classifier._modules.items():
                if isinstance(module, torch.nn.Linear):
                    filters = filters + module.out_features
        return filters

    def get_candidates_to_prune(self, num_filters_to_prune):
        self.prunner.reset()

        # self.train_epoch(rank_filters=True)
        self.calculate_pruning_score()
        # training 하면서 동시에 hook 써서 후보 찾기 #
        # (각 layer 마다 compute_rank 안에서 계산되어서 self.filter_ranks list에 저장이 된다.

        self.prunner.normalize_ranks_per_layer()  # Normalization

        filters_to_prune = self.prunner.get_pruned_filters_parallel(num_filters_to_prune)
        return filters_to_prune

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
        self.save_loss = False

        # Get the accuracy before prunning
        self.niter = 0
        self.temp = 0
        self.test_only(test_train=False)
        self.model.train()

        # Make sure all the layers are trainable
        # for param in self.model.features.parameters():
        #     param.requires_grad = True
        self.model.features_learning(True)

        number_of_filters = self.total_num_filters()
        num_filters_to_prune_per_iteration = int(
            number_of_filters * self.args.pr_step)  # 0.05 (5%) -> 0.01 (1%) temporally
        iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)
        iterations = max(int(iterations * self.args.total_pr), 1)  # up to 80% # fix bug not prune for iter=0

        for kk in range(1, iterations + 1):
            tqdm.tqdm.write('-' * 20+f"\nRanking filters.. step {kk}")
            self.niter += 1
            prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iteration)
            # prune_targets: 잘라야 할 filter들의 1) layer number, 2) filter number가 넘어옴
            layers_prunned = {}
            for layer_index, filter_index in prune_targets.items():
                layers_prunned[layer_index]=len(filter_index)
            tqdm.tqdm.write(f"Layers that will be prunned {layers_prunned}")
            tqdm.tqdm.write("Prunning filters.. ")
            # t=time.time()
            model = self.model.cpu()  # only prune in cpu
            for layer_index, prune_idx_list in prune_targets.items():
                model = prune_layer(model,layer_index,prune_idx_list,cuda_flag=False)
            self.model = model.cuda() if self.args.cuda else model
            # print(f'using {time.time()-t}')
            message = str(100 * float(self.total_num_filters()) / number_of_filters) + "%"
            tqdm.tqdm.write(f"Filters remained {str(message)}")
            if self.args.fine_tune:
                tqdm.tqdm.write("Fine tuning to recover from prunning iteration.")
                if hasattr(self.args,'fine_tune_conv') :
                    self.model.features_learning(self.args.fine_tune_conv)
                if hasattr(self.args,'fine_tune_fc') :
                    self.model.classifier_learning(self.args.fine_tune_fc)
                self.train_only(epoches=self.args.epochs)
            acc = self.test_only(test_train=False)
            if hasattr(self.args,'log_name') and self.args.log_name:
                log_name = self.args.log_name
                log_dir = os.path.dirname(log_name)
                if not os.path.exists(log_dir):
                    os.mkdir(log_dir)
                with open(log_name,'a') as logfile:
                    data=f'method:{self.args.method_type}-seed:{self.args.seed}-' \
                         f'prune_train:{self.args.prune_train}-fine_tune:{self.args.fine_tune}-' \
                         f'stage:{kk}-' \
                         f'prunned:{layers_prunned}-' \
                         f'Accuracy:{acc}' \
                         f'\n'
                    logfile.write(data)


        # print("Finished. Going to fine tune the model a bit more")
        # self.niter += 1
        # self.train(optimizer, epoches=5)
    def showlrp(self):
        # unfinished
        plt_size = 5
        for x_batch, y_batch in self.train_loader:
            x_batch = x_batch[:plt_size]
            y_batch = y_batch[:plt_size]
            if self.args.cuda:
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
            x_batch, y_batch = Variable(x_batch), Variable(y_batch)
            self.model.zero_grad()
            if self.args.relevance:
                if not hasattr(self.args,'sglrp') or not self.args.sglrp:
                    output = self.prunner.forward_lrp(x_batch)
                    if self.args.data_type == 'cifar10_raw':
                        cls_count = 10
                    elif self.args.data_type == 'cifar100_raw':
                        cls_count = 100
                    R = torch.nn.functional.one_hot(y_batch,cls_count)
                    # R = torch.zeros_like(output)
                    # for ii in range(len(y_batch)):
                    #     R[ii, y_batch[ii]] = 1.0

                    R = self.prunner.backward_lrp(R.data)

                else:
                    raise ValueError('!!')

            else:
                raise ValueError(f'only lrp valid, now {self.args.method_type}')

            # unfinished
            # image process: x, R
            img = x_batch.cpu()
            img = img*self.train_loader.dataset.std+self.train_loader.dataset.mean
            img = img.numpy()
            img = img.transpose((0,2,3,1))
            heatmap = R.cpu().numpy().transpose((0,2,3,1))
            heatmap = heatmap.sum(3)
            heatmap[heatmap<0]=0
            heatmap = (heatmap-heatmap.min(0))/(heatmap.max(0)-heatmap.min(0))

            plt.figure()
            for i in range(5):
                plt.subplot(2,plt_size,i+1)
                plt.imshow(img[i])
                plt.subplot(2,plt_size,i+1+plt_size)
                plt.imshow(heatmap[i])

            plt.show()
