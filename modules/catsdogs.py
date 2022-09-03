import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, ConcatDataset
import os
from torchvision import transforms

# dataset support for cat dog


class CatOrDogDataset(Dataset):
    def __init__(self, file_list, dir, mode='train', transform=None):
        self.file_list = file_list
        self.dir = dir
        self.mode = mode
        self.transform = transform
        if self.mode == 'train':
            if 'dog' in self.file_list[0]:
                self.label = 1
            else:
                self.label = 0

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.dir, self.file_list[idx]))
        if self.transform:
            img = self.transform(img)
        # if self.mode == 'train':
        #     img = img.numpy()
        #     return img.astype('float32'), self.label
        # else:
        #     img = img.numpy()
        #     return img.astype('float32'), self.file_list[idx]
        if self.mode == 'train':
            return img, self.label
        else:
            return img, self.file_list[idx]

def CatsDogs(root=r'F:\DataSet\dogs-vs-cats',prune_from_train=True, samples=None,transform=False):
    source_dir = os.path.join(root,'train')
    source_files = os.listdir(source_dir)
    # test_dir = './test1'
    # test_files = os.listdir(test_dir)
    cat_files = [tf for tf in source_files if 'cat' in tf]
    dog_files = [tf for tf in source_files if 'dog' in tf]

    if transform:
        train_transform = transforms.Compose([
            transforms.Resize(256),
            # transforms.ColorJitter(),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            # transforms.Resize(128),
            transforms.ToTensor()
        ])
    else:
        train_transform = transforms.Compose([transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

    cats_train = CatOrDogDataset(cat_files, source_dir, transform=train_transform)
    dogs_train = CatOrDogDataset(dog_files, source_dir, transform=train_transform)
    catdogs_train = ConcatDataset([cats_train, dogs_train])

    return catdogs_train




def get_catsdogs(root=r'F:\DataSet\dogs-vs-cats',prune_from_train=True, samples=None):
    source_dir = os.path.join(root,'train')
    source_files = os.listdir(source_dir)
    # test_dir = './test1'
    # test_files = os.listdir(test_dir)
    cat_files = [tf for tf in source_files if 'cat' in tf]
    dog_files = [tf for tf in source_files if 'dog' in tf]

    test_split = 0.3
    cat_train_len = int(len(cat_files)*(1-test_split))
    cat_train_files = cat_files[:cat_train_len]
    dog_train_files = dog_files[:cat_train_len]
    cat_test_files = cat_files[cat_train_len:]
    dog_test_files = dog_files[cat_train_len:]


    mean = torch.tensor([0.4883,0.4551,0.4170]).reshape(3,1,1)
    std = torch.tensor([0.2597,0.2531,0.2557]).reshape(3,1,1)

    train_transform = transforms.Compose([
        transforms.Resize((256,256)),
        # transforms.ColorJitter(),
        transforms.RandomCrop((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    common_transform = transforms.Compose([transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])

    cats_train = CatOrDogDataset(cat_train_files, source_dir, transform=train_transform)
    dogs_train = CatOrDogDataset(dog_train_files, source_dir, transform=train_transform)
    catdogs_train = ConcatDataset([cats_train, dogs_train])

    cats_test = CatOrDogDataset(cat_test_files, source_dir, transform=common_transform)
    dogs_test = CatOrDogDataset(dog_test_files, source_dir, transform=common_transform)
    catdogs_test= ConcatDataset([cats_test, dogs_test])

    if not samples:
        catdogs_prune = catdogs_train if prune_from_train else catdogs_test
    else:
        cat_files_prune, dog_files_prune = (cat_train_files,dog_train_files) \
            if prune_from_train else (cat_test_files,dog_test_files)
        cat_files_prune, dog_files_prune = np.array(cat_files_prune), np.array(dog_files_prune)
        cat_rand = np.random.choice(range(len(cat_files_prune)), samples*2)
        dog_rand = np.random.choice(range(len(cat_files_prune)), samples*2)
        cat_files_prune, dog_files_prune = cat_files_prune[cat_rand], dog_files_prune[dog_rand]
        cats_prune = CatOrDogDataset(cat_files_prune, source_dir, transform=common_transform)
        dogs_prune = CatOrDogDataset(dog_files_prune, source_dir, transform=common_transform)
        catdogs_prune = ConcatDataset([cats_prune, dogs_prune])


    return catdogs_train, catdogs_prune, catdogs_test




# class CatsDogs(Dataset):
#     def __init__(self, root=r'F:\DataSet\dogs-vs-cats', train=True,
#                  transform=None, samples=False):
#         if train:
#             self.dir = os.path.join(root,'train')
#         else:
#             self.dir = os.path.join(root, 'test1')
#
#         self.data, self.label = load_data(partition)
#         self.num_points = num_points
#         self.partition = partition
#         self.gaussian_noise = gaussian_noise
#         self.unseen = unseen
#         self.label = self.label.squeeze()
#         self.factor = factor
#         if self.unseen:
#             ######## simulate testing on first 20 categories while training on last 20 categories
#             if self.partition == 'test':
#                 self.data = self.data[self.label>=20]
#                 self.label = self.label[self.label>=20]
#             elif self.partition == 'train':
#                 self.data = self.data[self.label<20]
#                 self.label = self.label[self.label<20]
#
#     def __getitem__(self, item):
#         pointcloud = self.data[item][:self.num_points]          # 核心代码，就是用item取出的数据
#         if self.gaussian_noise:
#             pointcloud = jitter_pointcloud(pointcloud)
#         if self.partition != 'train':
#             np.random.seed(item)
#         anglex = np.random.uniform() * np.pi / self.factor
#         angley = np.random.uniform() * np.pi / self.factor
#         anglez = np.random.uniform() * np.pi / self.factor
#
#         cosx = np.cos(anglex)
#         cosy = np.cos(angley)
#         cosz = np.cos(anglez)
#         sinx = np.sin(anglex)
#         siny = np.sin(angley)
#         sinz = np.sin(anglez)
#         Rx = np.array([[1, 0, 0],
#                         [0, cosx, -sinx],
#                         [0, sinx, cosx]])
#         Ry = np.array([[cosy, 0, siny],
#                         [0, 1, 0],
#                         [-siny, 0, cosy]])
#         Rz = np.array([[cosz, -sinz, 0],
#                         [sinz, cosz, 0],
#                         [0, 0, 1]])
#         R_ab = Rx.dot(Ry).dot(Rz)
#         R_ba = R_ab.T
#         translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
#                                    np.random.uniform(-0.5, 0.5)])
#         translation_ba = -R_ba.dot(translation_ab)
#
#         pointcloud1 = pointcloud.T
#
#         rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
#         pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)
#
#         euler_ab = np.asarray([anglez, angley, anglex])
#         euler_ba = -euler_ab[::-1]
#
#         pointcloud1 = np.random.permutation(pointcloud1.T).T
#         pointcloud2 = np.random.permutation(pointcloud2.T).T
#         print(item)
#         print(pointcloud1.shape)
#         return pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), \
#                translation_ab.astype('float32'), R_ba.astype('float32'), translation_ba.astype('float32'), \
#                euler_ab.astype('float32'), euler_ba.astype('float32')
#
#     def __len__(self):
#         return self.data.shape[0]       # 给item一个范围
