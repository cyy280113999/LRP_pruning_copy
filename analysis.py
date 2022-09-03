import os

import re
import json
from itertools import product

import numpy as np
import matplotlib.pyplot as plt

filter_config_dict = {
    # 'alexnet': ({0: 0, 1: 3, 2: 6, 3: 8, 4: 10},
    #             [64, 192, 384, 256, 256]),
    'vgg16': ({0: 0, 1: 2, 2: 5, 3: 7, 4: 10, 5: 12, 6: 14, 7: 17, 8: 19, 9: 21, 10: 24, 11: 26, 12: 28, 13:32, 14:35},
              [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 4096, 4096])
}

map_PlotIdx_to_ModuleIdx, filters_total = filter_config_dict['vgg16']
filters_total = np.array(filters_total)
def countFilters(filter_step_list):
    map_to_conv_layer_index = {x[1]: x[0] for x in map_PlotIdx_to_ModuleIdx.items()}

    # set to total num
    filter_remain_iter = filters_total.copy()
    filter_remain_of_step = np.zeros((len(filter_step_list), len(map_PlotIdx_to_ModuleIdx)), dtype=float)
    for step, filter_cut in enumerate(filter_step_list):
        # for pruning_layer, pruning_num in filter_cut:  # for tuple
        filter_cut=eval(filter_cut)
        for pruning_layer, pruning_num in filter_cut.items():  # for dict
            filter_remain_iter[map_to_conv_layer_index[pruning_layer]] -= pruning_num
        filter_remain_of_step[step] = (filter_remain_iter.copy() / filters_total)
    return filter_remain_of_step

if __name__ == '__main__':

    with open('log.txt','r') as f:
        data = f.readlines()
    data = [l[:-1].split('-') for l in data]  # remove '\n'
    meta=\
    ['method','seed','prune_train','fine_tune','stage','prunned','Accuracy']
    [ 0      ,  1   ,     2       ,    3      ,   4   ,   5     ,   6   ]
    [str,      int,   bool,         bool,       int,    dict,     float]
    # meta_map = {i:j for i,j in zip(list(range(len(meta))),meta_type)}
    for i,line in enumerate(data):
        new_line=[]
        for j, pair in enumerate(line):
            after=pair[len(meta[j])+1:]
            new_line.append(after)
        data[i]=new_line
    data = np.array(data)


    # split by experiment
    seeds=np.unique(data[:,1])
    methods = np.unique(data[:,0])
    # -- always prune train!
    fts=np.unique(data[:,3])
    prs = np.unique(data[:,2])
    stages=np.unique(data[:,4])  # range(20)

    # make data by reduce stage
    for pr in prs:
        pr_data=data[data[:,2]==pr]
        for ft in fts:
            ft_data = pr_data[pr_data[:,3]==ft]
            method_acc_contrast=[]
            for method in methods:
                method_data=ft_data[ft_data[:,0]==method]
                # each seed may miss stage, create enough void list
                expr_filters=[[]for i in range(len(stages))]
                expr_acc=[[]for i in range(len(stages))]
                for seed in seeds:
                    seed_data = method_data[method_data[:,1]==seed]
                    seed_filters_remain=countFilters(seed_data[:,5])
                    seed_acc=seed_data[:, 6].astype(float)/100
                    for i,(stage_f,stage_acc) in enumerate(zip(seed_filters_remain,seed_acc)):
                        expr_filters[i].append(stage_f)
                        expr_acc[i].append(stage_acc)
                for i,(f,acc) in enumerate(zip(expr_filters,expr_acc)):
                    if f:
                        expr_filters[i]=np.array(f).mean(0)
                    if acc:
                        expr_acc[i]=np.array(acc if acc else [0]).mean(0)
                # remove miss stage
                expr_filters=[l for l in expr_filters if not isinstance(l,list)]
                expr_acc = np.array([l for l in expr_acc if l])
                method_acc_contrast.append(expr_acc)
                fig=plt.figure()
                for line in expr_filters:
                    plt.plot(line)
                    plt.xlim(0,14)
                    plt.ylim(0,1)
                plt.title(f'{method} filter {"ft"if ft=="True" else""}')
                fig.show()


            fig = plt.figure()
            for line,method in zip(method_acc_contrast,methods):
                plt.plot(line,label=method)
                plt.xlim(0, 20)
                plt.ylim(0.5, 1)
            plt.legend()
            plt.title(f'acc {"ft" if ft=="True" else ""}')
            fig.show()


                # print('hello')


        # avg over seed





    # method_data = data[data[:,1]==seeds[0]]
















