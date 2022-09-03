
import matplotlib.pyplot as plt
import re
import json
def main():
    cri_list = ['lrp','grad']
    fine_tune=False
    model = 'vgg16'
    file_name_dict = {
        'lrp':f'pruning_{model}_cifar_lrp_{"n"if not fine_tune else ""}ft.txt',
        'grad':f'pruning_{model}_cifar_grad_{"n"if not fine_tune else ""}ft.txt'
    }
    # file_name_list = ['pruning_alexnet_cifar_lrp_ft.txt','pruning_alexnet_cifar_grad_ft.txt']

    acc_cri_dict = {}
    filter_cri_dict = {}

    for cri in cri_list:
        with open(file_name_dict[cri]) as f:
            all_lines = f.read().split('\n')
            # all_lines = all_lines[:]
            step=0
            filter_step_list = [[]] # 0-18
            acc_step_list = [[]]
            for line in all_lines:
                if re.search('Ranking filters',line):
                    step+=1
                    filter_step_list.append([])
                    acc_step_list.append([])
                elif re.search('Test set: Average loss:',line):
                    acc_str = re.search('Accuracy: [0-9]+/10000 \([0-9]+%\)',line).group()
                    acc = re.findall('[0-9]+',acc_str)[2]
                    acc_step_list[step].append(float(acc)/100)
                elif re.search('Layers that will be prunned',line):
                    filter_str = re.search('\{.*\}',line).group()
                    filter_of_layer = filter_str[1:-1]
                    filter_of_layer_list = filter_of_layer.split(',')
                    filter_step_list[step]=[list(map(int,layer_data.split(':'))) for layer_data in filter_of_layer_list]
                elif re.search('Finished. Going to fine tune the model a bit more',line):
                    break
        filter_cri_dict[cri] = filter_step_list
        acc_cri_dict[cri] = acc_step_list

            # filter_cri_list += [re.search('\{.*\}',filters).group() for filters in all_lines if re.search('Layers that will be prunned',filters)]

    # print(filter_cri_list)
    # print(acc_cri_list)
    if fine_tune:
        plt.figure()
        plt.plot([step[0] for step in acc_cri_dict['lrp']],color='r',label='lrp')
        plt.plot([step[0] for step in acc_cri_dict['grad']],color='g',label='grad')
        plt.title('lrp vs grad before')
        plt.grid(True)
        plt.legend()
        plt.figure()
        plt.plot([step[-1] for step in acc_cri_dict['lrp']],color='r',label='lrp')
        plt.plot([step[-1] for step in acc_cri_dict['grad']],color='g',label='grad')
        plt.grid(True)
        plt.title('lrp vs grad after')
        plt.legend()
    else:
        plt.figure()
        plt.plot([step[-1] for step in acc_cri_dict['lrp']],color='r',label='lrp')
        plt.plot([step[-1] for step in acc_cri_dict['grad']],color='g',label='grad')
        plt.grid(True)
        plt.title('lrp vs grad')
        plt.legend()

    # 剩余神经元数量可视化
    filter_config_dict={
        'alexnet':({0:0, 1:3, 2:6, 3:8, 4:10},
                   [64, 192, 384, 256, 256]),
        'vgg16':({0:0,1:2,2:5,3:7,4:10,5:12,6:14,7:17,8:19,9:21,10:24,11:26,12:28},
                 [64,64,128,128,256,256,256,512,512,512,512,512,512])
    }
    '''VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)   '''

    filter_map_to_layer,filters_total = filter_config_dict[model]
    filter_map_to_data = {x[1]:x[0] for x in filter_map_to_layer.items()}

    plt.figure()
    filter_remain = filters_total.copy()
    filter_remain_step_list = []
    for step in filter_cri_dict['lrp']:
        for pruning in step:
            filter_remain[filter_map_to_data[pruning[0]]]-=pruning[1]
        filter_remain_step_list.append(filter_remain.copy())
    for step in filter_remain_step_list:
        plt.plot([x[0]/x[1] for x in zip(step,filters_total)])

    plt.title('lrp filter remained')

    plt.figure()
    filter_remain = filters_total.copy()
    filter_remain_step_list = []
    for step in filter_cri_dict['grad']:
        for pruning in step:
            filter_remain[filter_map_to_data[pruning[0]]] -= pruning[1]
        filter_remain_step_list.append(filter_remain.copy())
    for step in filter_remain_step_list:
        plt.plot([x[0] / x[1] for x in zip(step, filters_total)])

    plt.title('grad filter remained')

    plt.show()



if __name__ == '__main__':
    main()