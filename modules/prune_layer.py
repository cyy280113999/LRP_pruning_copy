import torch
from torch.nn.utils import prune as torch_prune
import numpy as np
from collections import OrderedDict


def prune_conv_layer(model, layer_index, filter_index, criterion='lrp', cuda_flag=False):
    ''' input parameters
    1. model: 현재 모델
    2. layer_index: 자르고자 하는 layer index
    3. filter_index: 자르고자 하는 layer의 filter index
    '''

    conv = dict(model.named_modules())[layer_index]

    if not hasattr(conv, "output_mask"):
        # Instantiate output mask tensor of shape (num_output_channels, )
        conv.output_mask = torch.ones(conv.weight.shape[0])

    # Make sure the filter was not pruned before
    assert conv.output_mask[filter_index] != 0

    conv.output_mask[filter_index] = 0

    mask_weight = conv.output_mask.view(-1, 1, 1, 1).expand_as(
        conv.weight)
    torch_prune.custom_from_mask(conv, "weight", mask_weight)

    if conv.bias is not None:
        mask_bias = conv.output_mask
        torch_prune.custom_from_mask(conv, "bias", mask_bias)

    if cuda_flag:
        conv.weight = conv.weight.cuda()
        # conv.module.bias = conv.module.bias.cuda()

    return model


def replace_layers(model, i, indexes, layers):
    if i in indexes:
        return layers[indexes.index(i)]
    return model[i]

# prune layer generally , require one layer and filters
# decide prune parallely by num of filters
 # only prune in cpu
def prune_layer(model, layer_index, filter_indexs, cuda_flag=False):
    # model must be structure of [features,classifier]
    # input layer_index is index from sequence(features,classifier)
    # input filter_indexs is list
    features_layers = len(model.features._modules)
    if not isinstance(filter_indexs, list):
        if layer_index < features_layers:
            model = prune_conv_layer_sequential(model, layer_index, filter_indexs, cuda_flag)
        else:
            layer_index -= features_layers
            model = prune_fc_layer_sequential(model, layer_index, filter_indexs, cuda_flag)
    else:
        if layer_index < features_layers:
            model = prune_conv_layer_sequential_parallel(model, layer_index, filter_indexs, cuda_flag)
        else:
            layer_index -= features_layers
            model = prune_fc_layer_sequential_parallel(model, layer_index, filter_indexs, cuda_flag)
    return model


def prune_conv_layer_sequential(model, layer_index, filter_index, cuda_flag=False):
    ''' input parameters
    1. model: 현재 모델
    2. layer_index: 자르고자 하는 layer index
    3. filter_index: 자르고자 하는 layer의 filter index
    '''
    # _, conv = model.features._modules.items()[layer_index]
    _, conv = list(model.features._modules.items())[layer_index]  # 해당 layer를 우선 끄집어 온다.
    next_conv = None
    offset = 1

    while layer_index + offset < len(model.features._modules.items()):  # 전체 network의 layer 수보다 클때까지 while문 반복
        # res =  model.features._modules.items()[layer_index+offset]
        res = list(model.features._modules.items())[layer_index + offset]
        if isinstance(res[1], torch.nn.modules.conv.Conv2d):  # 현재 layer를 기준으로 다음 layer가 conv layer이냐?
            next_name, next_conv = res
            break
        offset = offset + 1

    # 새로운 conv_layer(new_conv)를 다시 생성시킨다.
    # output 쪽의 갯수를 하나 줄여준다.
    new_conv = \
        torch.nn.Conv2d(in_channels=conv.in_channels,
                        out_channels=conv.out_channels - 1,
                        kernel_size=conv.kernel_size,
                        stride=conv.stride,
                        padding=conv.padding,
                        dilation=conv.dilation,
                        groups=conv.groups,
                        bias=True)

    old_weights = conv.weight.data.cpu().numpy()
    new_weights = new_conv.weight.data.cpu().numpy()

    # for p in new_conv.parameters():
    #     p.requires_grad = False

    # weight는 해당 filter를 제외하고 총 갯수 - 1 를 넣어준다.
    new_weights[: filter_index, :, :, :] = old_weights[: filter_index, :, :, :]
    new_weights[filter_index:, :, :, :] = old_weights[filter_index + 1:, :, :, :]
    new_conv.weight.data = torch.from_numpy(new_weights).cuda() if cuda_flag else torch.from_numpy(new_weights)

    # bias도 해당 filter number의 값을 제외하고 총 갯수 -1를 넣어준다.
    bias_numpy = conv.bias.data.cpu().numpy()
    bias = np.zeros(shape=(bias_numpy.shape[0] - 1), dtype=np.float32)
    bias[:filter_index] = bias_numpy[:filter_index]
    bias[filter_index:] = bias_numpy[filter_index + 1:]
    new_conv.bias.data = torch.from_numpy(bias).cuda() if cuda_flag else torch.from_numpy(bias)

    # 다음 conv layer도 받는 쪽의 layer 갯수를 줄여준다.
    if not next_conv is None:
        next_new_conv = \
            torch.nn.Conv2d(in_channels=next_conv.in_channels - 1,
                            out_channels=next_conv.out_channels,
                            kernel_size=next_conv.kernel_size,
                            stride=next_conv.stride,
                            padding=next_conv.padding,
                            dilation=next_conv.dilation,
                            groups=next_conv.groups,
                            bias=True)

        old_weights = next_conv.weight.data.cpu().numpy()
        new_weights = next_new_conv.weight.data.cpu().numpy()

        # for p in next_new_conv.parameters():
        #     p.requires_grad = False

        new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
        new_weights[:, filter_index:, :, :] = old_weights[:, filter_index + 1:, :, :]
        next_new_conv.weight.data = torch.from_numpy(new_weights).cuda() if cuda_flag else torch.from_numpy(new_weights)

        next_new_conv.bias.data = next_conv.bias.data

    if not next_conv is None:
        features = torch.nn.Sequential(
            *(replace_layers(model.features, i, [layer_index, layer_index + offset],
                             [new_conv, next_new_conv]) for i, _ in enumerate(model.features)))
        del model.features
        del conv

        model.features = features

    else:
        # Prunning the last conv layer. This affects the first linear layer of the classifier.
        model.features = torch.nn.Sequential(
            *(replace_layers(model.features, i, [layer_index], \
                             [new_conv]) for i, _ in enumerate(model.features)))
        layer_index = 0
        old_linear_layer = None
        for _, module in model.classifier._modules.items():
            if isinstance(module, torch.nn.Linear):
                old_linear_layer = module
                break
            layer_index = layer_index + 1

        if old_linear_layer is None:
            raise BaseException("No linear laye found in classifier")
        params_per_input_channel = int(old_linear_layer.in_features / conv.out_channels)

        new_linear_layer = \
            torch.nn.Linear(old_linear_layer.in_features - params_per_input_channel,
                            old_linear_layer.out_features)

        old_weights = old_linear_layer.weight.data.cpu().numpy()
        new_weights = new_linear_layer.weight.data.cpu().numpy()

        new_weights[:, : filter_index * params_per_input_channel] = \
            old_weights[:, : filter_index * params_per_input_channel]
        new_weights[:, filter_index * params_per_input_channel:] = \
            old_weights[:, (filter_index + 1) * params_per_input_channel:]

        new_linear_layer.bias.data = old_linear_layer.bias.data

        new_linear_layer.weight.data = torch.from_numpy(new_weights).cuda() if cuda_flag else torch.from_numpy(
            new_weights)

        classifier = torch.nn.Sequential(
            *(replace_layers(model.classifier, i, [layer_index], \
                             [new_linear_layer]) for i, _ in enumerate(model.classifier)))

        del model.classifier
        del next_conv
        del conv
        model.classifier = classifier

    return model

# prune a conv layer in model by list of filters(channels)
def prune_conv_layer_sequential_parallel(model, layer_index, filter_indexs, cuda_flag=False):
    _, conv = list(model.features._modules.items())[layer_index]
    new_conv = torch.nn.Conv2d(in_channels=conv.in_channels,
                               out_channels=conv.out_channels - len(filter_indexs),
                               kernel_size=conv.kernel_size,
                               stride=conv.stride,
                               padding=conv.padding,
                               dilation=conv.dilation,
                               groups=conv.groups,
                               bias=True)
    assert new_conv.out_channels, f'all removed in layer{layer_index}'

    old_weights = conv.weight.data
    new_weights = new_conv.weight.data
    old_bias = conv.bias.data
    new_bias = new_conv.bias.data

    filter_indexs.sort()
    copyFromRange_list = []
    for ii, prune_idx in enumerate(filter_indexs):
        if ii == 0 and prune_idx != 0:
            copyFromRange_list.append((0, prune_idx))  # right open
        elif ii != 0:
            last_pruned = filter_indexs[ii - 1]
            if prune_idx != last_pruned + 1:
                copyFromRange_list.append((last_pruned + 1, prune_idx))
    last_pruned = filter_indexs[-1]
    prune_idx = old_weights.shape[0]
    if prune_idx != last_pruned + 1:
        copyFromRange_list.append((last_pruned + 1, prune_idx))

    # assert copyFromRange_list, f'you cut {layer_index},{filter_indexs}\n,' \
    #                            f'but has:{conv.out_channels}'

    copyToStart_idx = 0
    try:
        for (copyFromLeft_idx, copyFromRight_idx) in copyFromRange_list:
            copyToEnd_idx = copyToStart_idx + copyFromRight_idx - copyFromLeft_idx
            new_weights[copyToStart_idx:copyToEnd_idx] = old_weights[copyFromLeft_idx:copyFromRight_idx]
            new_bias[copyToStart_idx:copyToEnd_idx] = old_bias[copyFromLeft_idx:copyFromRight_idx]
            copyToStart_idx = copyToEnd_idx
    except Exception as e:
        print(e)

    # find out next parameterized layer is conv or linear
    next_conv = None
    offset = 1

    while layer_index + offset < len(model.features._modules.items()):
        # res =  model.features._modules.items()[layer_index+offset]
        res = list(model.features._modules.items())[layer_index + offset]
        if isinstance(res[1], torch.nn.modules.conv.Conv2d):
            next_name, next_conv = res
            break
        offset = offset + 1
    if next_conv is not None:
        next_new_conv = torch.nn.Conv2d(in_channels=next_conv.in_channels - len(filter_indexs),
                                        out_channels=next_conv.out_channels,
                                        kernel_size=next_conv.kernel_size,
                                        stride=next_conv.stride,
                                        padding=next_conv.padding,
                                        dilation=next_conv.dilation,
                                        groups=next_conv.groups,
                                        bias=True)

        old_weights = next_conv.weight.data
        new_weights = next_new_conv.weight.data

        copyToStart_idx = 0
        for (copyFromLeft_idx, copyFromRight_idx) in copyFromRange_list:
            copyToEnd_idx = copyToStart_idx + copyFromRight_idx - copyFromLeft_idx
            new_weights[:, copyToStart_idx:copyToEnd_idx] = old_weights[:, copyFromLeft_idx:copyFromRight_idx]
            copyToStart_idx = copyToEnd_idx
        next_new_conv.bias.data = next_conv.bias.data

        features = torch.nn.Sequential(
            *(replace_layers(model.features, i, [layer_index, layer_index + offset],
                             [new_conv, next_new_conv]) for i, _ in enumerate(model.features)))
        del model.features
        del conv

        model.features = features

    else:
        model.features = torch.nn.Sequential(
            *(replace_layers(model.features, i, [layer_index],
                             [new_conv]) for i, _ in enumerate(model.features)))
        layer_index = 0
        old_lin = None
        for _, module in model.classifier._modules.items():
            if isinstance(module, torch.nn.Linear):
                old_lin = module
                break
            layer_index = layer_index + 1
        if old_lin is None: raise Exception("No linear layer found in classifier!!!!")

        params_per_input_channel = int(old_lin.in_features / conv.out_channels)

        new_lin = torch.nn.Linear(old_lin.in_features - params_per_input_channel * len(filter_indexs),
                                  old_lin.out_features)

        old_weights = old_lin.weight.data
        new_weights = new_lin.weight.data

        copyToStart_idx = 0
        for (copyFromLeft_idx, copyFromRight_idx) in copyFromRange_list:
            copyToEnd_idx = copyToStart_idx + copyFromRight_idx - copyFromLeft_idx
            new_weights[:, copyToStart_idx * params_per_input_channel:copyToEnd_idx * params_per_input_channel] = \
                old_weights[:, copyFromLeft_idx * params_per_input_channel:copyFromRight_idx * params_per_input_channel]
            copyToStart_idx = copyToEnd_idx

        new_lin.bias.data = old_lin.bias.data

        classifier = torch.nn.Sequential(
            *(replace_layers(model.classifier, i, [layer_index],
                             [new_lin]) for i, _ in enumerate(model.classifier)))

        del model.classifier
        del next_conv
        del conv
        model.classifier = classifier

    return model


if __name__ == '__main__':
    # pass
    model = torch.nn.Module()
    model.features = torch.nn.Sequential(torch.nn.Conv2d(10, 20, 3),
                                         torch.nn.Conv2d(20, 5, 3))
    filter_indexs = list(range(1, 5)) + list(range(11, 15))
    prune_conv_layer_sequential(model, 0, filter_indexs, False)
    model.features = torch.nn.Sequential(torch.nn.Conv2d(10, 20, 3))
    model.classifier = torch.nn.Sequential(torch.nn.Linear(20 * 10, 5))
    prune_conv_layer_sequential(model, 0, filter_indexs, False)

# prune fc by one layer one neuron
def prune_fc_layer_sequential(model, layer_index, filter_index, cuda_flag=False):
    _, dense = list(model.classifier._modules.items())[layer_index]  # 해당 layer를 우선 끄집어 온다.

    next_dense = None
    offset = 1
    while layer_index + offset < len(model.classifier._modules.items()):  # 전체 network의 layer 수보다 클때까지 while문 반복
        res = list(model.classifier._modules.items())[layer_index + offset]
        if isinstance(res[1], torch.nn.modules.linear.Linear):  # 현재 layer를 기준으로 다음 layer가 dense layer이냐?
            next_name, next_dense = res
            break
        offset = offset + 1

    new_dense = torch.nn.Linear(in_features=dense.in_features,
                                out_features=dense.out_features - 1,
                                bias=True)

    if cuda_flag: new_dense = new_dense.cuda()

    old_weights = dense.weight.data
    new_weights = new_dense.weight.data

    new_weights[: filter_index, :] = old_weights[: filter_index, :]
    new_weights[filter_index:, :] = old_weights[filter_index + 1:, :]
    # new_dense.weight.data = torch.from_numpy(new_weights).cuda() if cuda_flag else torch.from_numpy(new_weights)

    old_bias = dense.bias.data
    new_bias = new_dense.bias.data
    new_bias[:filter_index] = old_bias[:filter_index]
    new_bias[filter_index:] = old_bias[filter_index + 1:]
    # new_dense.bias.data = torch.from_numpy(new_bias).cuda() if cuda_flag else torch.from_numpy(new_bias)

    # 다음 conv layer도 받는 쪽의 layer 갯수를 줄여준다.
    if next_dense is not None:
        next_new_dense = \
            torch.nn.Linear(in_features=next_dense.in_features - 1,
                            out_features=next_dense.out_features,
                            bias=True)

        old_weights = next_dense.weight.data.cpu().numpy()
        new_weights = next_new_dense.weight.data.cpu().numpy()

        # for p in next_new_conv.parameters():
        #     p.requires_grad = False

        new_weights[:, : filter_index] = old_weights[:, : filter_index]
        new_weights[:, filter_index:] = old_weights[:, filter_index + 1:]
        next_new_dense.weight.data = torch.from_numpy(new_weights).cuda() if cuda_flag else torch.from_numpy(
            new_weights)

        next_new_dense.bias.data = next_dense.bias.data

    if not next_dense is None:
        network = torch.nn.Sequential(
            *(replace_layers(model.classifier, i, [layer_index, layer_index + offset],
                             [new_dense, next_new_dense]) for i, _ in enumerate(model.classifier)))
        del model.classifier
        del dense

        model.classifier = network
    else:
        network = torch.nn.Sequential(
            *(replace_layers(model.classifier, i, [layer_index],
                             [new_dense]) for i, _ in enumerate(model.classifier)))
        del model.classifier
        del dense

        model.classifier = network

    return model

# prune a fc layer in model by list of neurons
def prune_fc_layer_sequential_parallel(model, layer_index, neuron_indexs, cuda_flag=False):
    _, dense = list(model.classifier._modules.items())[layer_index]  # 해당 layer를 우선 끄집어 온다.

    new_dense = torch.nn.Linear(in_features=dense.in_features,
                                out_features=dense.out_features - len(neuron_indexs), bias=True)
    assert new_dense.out_features, f'all removed in layer{layer_index}'
    if cuda_flag: new_dense = new_dense.cuda()

    old_weights = dense.weight.data
    new_weights = new_dense.weight.data
    old_bias = dense.bias.data
    new_bias = new_dense.bias.data

    # ! can arise some bugs
    # get complementary interval by pruned neuron_indexs
    # copyFromRange_list = [(copy_left,copy_right)] # right excluded
    neuron_indexs.sort()
    copyFromRange_list = []
    for ii, prune_idx in enumerate(neuron_indexs):
        if ii == 0 and prune_idx != 0:
            copyFromRange_list.append((0, prune_idx))  # right open
        elif ii != 0:
            last_pruned = neuron_indexs[ii - 1]
            if prune_idx != last_pruned + 1:
                copyFromRange_list.append((last_pruned + 1, prune_idx))
    last_pruned = neuron_indexs[-1]
    prune_idx = old_weights.shape[0]
    if prune_idx != last_pruned + 1:
        copyFromRange_list.append((last_pruned + 1, prune_idx))

    # check data from up-side
    assert copyFromRange_list
    # for prune_idx in neuron_indexs:
    #     for left, right in copyFromRange_list:
    #         assert not left <= prune_idx < right

    # copy
    # new_array[(new interval)] = old_array[(copy_left,copy_right)]
    try:
        copyToStart_idx = 0
        for (copyFromLeft_idx, copyFromRight_idx) in copyFromRange_list:
            copyToEnd_idx = copyToStart_idx + copyFromRight_idx - copyFromLeft_idx
            new_weights[copyToStart_idx:copyToEnd_idx] = old_weights[copyFromLeft_idx:copyFromRight_idx]
            new_bias[copyToStart_idx:copyToEnd_idx] = old_bias[copyFromLeft_idx:copyFromRight_idx]
            copyToStart_idx = copyToEnd_idx
    except Exception as e:
        print(e)
    next_dense = None
    offset = 1
    while layer_index + offset < len(model.classifier._modules.items()):
        res = list(model.classifier._modules.items())[layer_index + offset]
        if isinstance(res[1], torch.nn.modules.linear.Linear):
            next_name, next_dense = res
            break
        offset = offset + 1

    if next_dense is not None:
        next_new_dense = \
            torch.nn.Linear(in_features=next_dense.in_features - len(neuron_indexs),
                            out_features=next_dense.out_features,
                            bias=True)

        old_weights = next_dense.weight.data
        new_weights = next_new_dense.weight.data
        copyToStart_idx = 0
        for (copyFromLeft_idx, copyFromRight_idx) in copyFromRange_list:
            copyToEnd_idx = copyToStart_idx + copyFromRight_idx - copyFromLeft_idx
            new_weights[:, copyToStart_idx:copyToEnd_idx] = old_weights[:, copyFromLeft_idx:copyFromRight_idx]
            copyToStart_idx = copyToEnd_idx

        next_new_dense.bias.data = next_dense.bias.data

    if next_dense is not None:
        network = torch.nn.Sequential(
            *(replace_layers(model.classifier, i, [layer_index, layer_index + offset],
                             [new_dense, next_new_dense]) for i, _ in enumerate(model.classifier)))
        del model.classifier
        del dense

        model.classifier = network
    else:
        network = torch.nn.Sequential(
            *(replace_layers(model.classifier, i, [layer_index],
                             [new_dense]) for i, _ in enumerate(model.classifier)))
        del model.classifier
        del dense

        model.classifier = network

    return model


if __name__ == '__main__':
    # pass
    model = torch.nn.Module()
    model.classifier = torch.nn.Sequential(torch.nn.Linear(10, 20),
                                           torch.nn.Linear(20, 5))
    filter_indexs = list(range(1, 5)) + list(range(11, 15))
    prune_fc_layer_sequential(model, 0, filter_indexs, False)

# prune last fc layer, make model to subclass. same as prune fc generally
def prune_to_classes(model, class_idxs, cuda_flag=False):
    assert model.classifier
    classifier = list(model.classifier._modules.items())
    name, fc = classifier[-1]
    assert isinstance(fc, torch.nn.Linear)

    new_fc = torch.nn.Linear(in_features=fc.in_features,
                             out_features=len(class_idxs),
                             bias=True)
    assert new_fc.out_features, f'kill all classes!'
    if cuda_flag:
        new_fc = new_fc.cuda()

    old_weights = fc.weight.data
    new_weights = new_fc.weight.data
    old_bias = fc.bias.data
    new_bias = new_fc.bias.data

    for new_idx, idx in enumerate(class_idxs):
        new_weights[new_idx] = old_weights[idx]
        new_bias[new_idx] = old_bias[idx]
    # new_fc.weight.data = torch.from_numpy(new_weights).cuda() if cuda_flag else torch.from_numpy(new_weights)
    # new_fc.bias.data = torch.from_numpy(new_bias).cuda() if cuda_flag else torch.from_numpy(new_bias)

    classifier = classifier[:-1] + [(name, new_fc)]
    model.classifier = torch.nn.Sequential(OrderedDict(classifier))

    return model
