import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
#
# new dataset pre-store GPU for some small dataset:CIFAR


def get_cifar10_raw(datapath=r'F:\DataSet\cifar10', class_idxs=None, samples=None):
    '''
    Get CIFAR10 dataset
    '''
    # mean = (0.4914, 0.4822, 0.4465)
    # std = (0.2471, 0.2435, 0.2616)
    # normalize = transforms.Normalize(mean=mean, std=std)
    # # Cifar-10 Dataset
    # train_dataset = datasets.CIFAR10(root=datapath,
    #                                  train=True,
    #                                  transform=transforms.Compose([
    #                                      transforms.ToTensor(),
    #                                      normalize
    #                                  ]),
    #                                  download=download)
    #
    # test_dataset = datasets.CIFAR10(root=datapath,
    #                                 train=False,
    #                                 transform=transforms.Compose([
    #                                     transforms.ToTensor(),
    #                                     normalize
    #                                 ]))
    # GPU无法数据增强，速度太慢
    DataSet = CUDA_CIFAR10
    train_dataset = DataSet(root=datapath,
                                 train=True,
                                 to_cuda=True,
                                 transform=transforms.Compose([
                                     # transforms.RandomPerspective(),
                                     # transforms.RandomGrayscale(),
                                     transforms.RandomHorizontalFlip(),
                                     # transforms.RandomRotation(15)
                                 ]),
                                 class_idxs=class_idxs)
    if not samples:
        prune_dataset = train_dataset
    else:
        prune_dataset = DataSet(root=datapath,
                                     train=True,
                                     to_cuda=True,
                                     class_idxs=class_idxs,
                                     samples=samples)

    test_dataset = DataSet(root=datapath,
                                train=False,
                                to_cuda=True,
                                class_idxs=class_idxs)

    return train_dataset, prune_dataset, test_dataset


def get_cifar100_raw(datapath=r'F:\DataSet\cifar100', class_idxs=None, samples=None):
    # mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    # std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
    # normalize = transforms.Normalize(mean=mean, std=std)
    # train_dataset = datasets.CIFAR100(root=datapath,
    #                                   train=True,
    #                                   transform=transforms.Compose([
    #                                       # transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
    #                                       transforms.ToTensor(),
    #                                       normalize,
    #                                       # transforms.RandomPerspective(),
    #                                       # transforms.RandomGrayscale(),
    #                                       transforms.RandomHorizontalFlip(),
    #                                       transforms.RandomRotation(15)
    #                                   ]),
    #                                   download=download)
    # test_dataset = datasets.CIFAR100(root=datapath,
    #                                  train=False,
    #                                  transform=transforms.Compose([
    #                                      transforms.ToTensor(),
    #                                      normalize
    #                                  ]),
    #                                  download=download)
    # GPU数据增强速度太慢: no rotate 22s vs rotate 60s
    DataSet = CUDA_CIFAR100
    train_dataset = DataSet(root=datapath,
                                 train=True,
                                 to_cuda=True,
                                 transform=transforms.Compose([
                                     # transforms.RandomPerspective(),
                                     # transforms.RandomGrayscale(),
                                     transforms.RandomHorizontalFlip(),
                                     # transforms.RandomRotation(15)
                                 ]),
                                  class_idxs=class_idxs)
    if not samples:
        prune_dataset = train_dataset
    else:
        prune_dataset = DataSet(root=datapath,
                                     train=True,
                                     to_cuda=True,
                                     class_idxs=class_idxs,
                                     samples=samples)
    test_dataset = DataSet(root=datapath,
                                train=False,
                                to_cuda=True,
                                class_idxs=class_idxs)
    return train_dataset, prune_dataset, test_dataset

class CUDA_CIFAR10(CIFAR10):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).reshape((3, 1, 1))
    std = torch.tensor([0.2471, 0.2435, 0.2616]).reshape((3, 1, 1))

    def __init__(
            self,
            root='F:\workspace\_Project\_DataSet\cifar10',
            train: bool = True,
            to_cuda: bool = False,
            half: bool = False,
            pre_transform=None,
            transform=None,
            target_transform=None,
            download: bool = False,
            class_idxs=None, samples=None):
        super().__init__(root, train, transform, target_transform, download=False)
        # 预处理 等价ToTensor Normalize
        # 1 int numpy --> float tensor
        # 2 - mean / std
        self.data = torch.tensor(self.data.transpose((0, 3, 1, 2)) / 255.0, dtype=torch.float)
        self.targets = torch.tensor(self.targets, dtype=torch.long)
        if class_idxs:
            data = None
            targets = None
            for new_cls_idx, class_idx in enumerate(class_idxs):
                position = np.where(self.targets.numpy() == class_idx)[0]
                if samples:
                    position = np.random.choice(position, samples)# fixed sample
                if data is None:
                    data = self.data[position]
                    targets = torch.ones(len(position),dtype=torch.long) * new_cls_idx
                else:
                    data = torch.vstack([data, self.data[position]])
                    targets = torch.cat([targets, torch.ones(len(position),dtype=torch.long) * new_cls_idx])
            self.data = data
            self.targets = targets

        self.data = (self.data - self.mean) / self.std
        if to_cuda:
            self.data = self.data.cuda()
            self.targets = self.targets.cuda()

        # if pre_transform is not None:
        #     self.data = self.data.astype("float32")
        #     for index in range(len(self)):
        #         """
        #         ToTensor的操作会检查数据类型是否为uint8, 如果是, 则除以255进行归一化, 这里data提前转为float,
        #         所以手动除以255.
        #         """
        #         self.data[index] = pre_transform(self.data[index] / 255.0).numpy().transpose((1, 2, 0))
        #         self.targets[index] = torch.Tensor([self.targets[index]]).squeeze_().long()
        #         if to_cuda:
        #             self.targets[index] = self.targets[index].cuda()
        #     self.data = torch.Tensor(self.data).permute((0, 3, 1, 2))
        #     if half:
        #         self.data:torch.Tensor = self.data.half()
        #     if to_cuda:
        #         self.data = self.data.cuda()

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class CUDA_CIFAR100(CIFAR100, CUDA_CIFAR10):
    mean = torch.tensor([0.5070751592371323, 0.48654887331495095, 0.4409178433670343]).reshape((3, 1, 1))
    std = torch.tensor([0.2673342858792401, 0.2564384629170883, 0.27615047132568404]).reshape((3, 1, 1))
