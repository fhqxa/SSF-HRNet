# code is based on https://github.com/katerakelly/pytorch-maml
import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import Sampler
from SSFR import SCR



def imshow(img):
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

class Rotate(object):
    def __init__(self, angle):
        self.angle = angle
    def __call__(self, x, mode="reflect"):
        x = x.rotate(self.angle)
        return x


def tired_imagenet_folders():
    train_folder = '/home/dj212/ZYQ/Datasets/tieredimagenet-deep/train'
    test_folder = '/home/dj212/ZYQ/Datasets/tieredimagenet-deep/val'

    metatrain_folders = [os.path.join(train_folder, coarse_label, label) \
                for coarse_label in os.listdir(train_folder) \
                if os.path.isdir(os.path.join(train_folder, coarse_label)) \
                for label in os.listdir(os.path.join(train_folder, coarse_label))
                ]
    metatest_folders = [os.path.join(test_folder, coarse_label, label) \
                for coarse_label in os.listdir(test_folder) \
                if os.path.isdir(os.path.join(test_folder, coarse_label)) \
                for label in os.listdir(os.path.join(test_folder, coarse_label))
                ]

    random.seed(1)
    random.shuffle(metatrain_folders)
    random.shuffle(metatest_folders)
    return metatrain_folders, metatest_folders

class TiredImagenetTask(object):

    def __init__(self, character_folders, num_classes, train_num, test_num):
        self.character_folders = character_folders
        self.num_classes = num_classes  # 5
        self.train_num = train_num  # 1训练图
        self.test_num = test_num  # 10查询图

        class_folders = random.sample(self.character_folders, self.num_classes)  # 随机抽取5个文件夹
        coarse_class_folders = np.unique([self.get_class(x) for x in class_folders])

# 生成粗类标签
        coarse_labels = np.array(range(len(coarse_class_folders)))
        coarse_labels = dict(zip(coarse_class_folders, coarse_labels))

        labels = np.array(range(len(class_folders)))
        labels = dict(zip(class_folders, labels))
        samples = dict()

        self.train_roots = []  # 每个情景用于训练的图的路径
        self.test_roots = []  # 每个情景用于测试的图的路径
        for c in class_folders:
            temp = [os.path.join(c, x) for x in os.listdir(c)]
            samples[c] = random.sample(temp, len(temp))
            random.shuffle(samples[c])

            self.train_roots += samples[c][:train_num]
            self.test_roots += samples[c][train_num:train_num+test_num]

        self.train_coarse_labels = [coarse_labels[self.get_coarse_class(x)] for x in self.train_roots]
        self.test_coarse_labels = [coarse_labels[self.get_coarse_class(x)] for x in self.test_roots]

        self.train_labels = [labels['/'+self.get_class(x)] for x in self.train_roots]
        self.test_labels = [labels['/'+self.get_class(x)] for x in self.test_roots]

    def get_class(self, sample):
        return os.path.join(*sample.split('/')[:-1])

    def get_coarse_class(self, sample):
        return os.path.join(*sample.split('/')[:-2])

class FewShotDataset(Dataset):

    def __init__(self, task, split='train', transform=None, target_transform=None):
        self.transform = transform  # Torch operations on the input image
        self.target_transform = target_transform
        self.task = task
        self.split = split
        self.image_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels
        self.coarse_labels = self.task.train_coarse_labels if self.split == 'train' else self.task.test_coarse_labels

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")

class TiredImagenet(FewShotDataset):

    def __init__(self, *args, **kwargs):
        super(TiredImagenet, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        image = Image.open(image_root)
        image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        coarse_label = self.coarse_labels[idx]

        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, coarse_label, label


class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pools
        of examples of size 'num_per_class' '''

    def __init__(self, num_per_class, num_cl, num_inst, shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):

        if self.shuffle:
            batch = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        else:
            batch = [[i+j*self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1


def get_tired_imagenet_data_loader(task, num_per_class=1, split='train', shuffle=False, rotation=0):
    mean = [120.39586422/255.0, 115.59361427/255.0, 104.54012653/255.0]
    std = [70.68188272/255.0, 68.27635443/255.0, 72.54505529/255.0]
    normalize = transforms.Normalize(mean=mean, std=std)

    dataset = TiredImagenet(task, split=split, transform=transforms.Compose([transforms.ToTensor(), transforms.Resize([84, 84]), normalize]))

    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.train_num, shuffle=shuffle)
    else:
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.test_num, shuffle=shuffle)
    loader = DataLoader(dataset, batch_size=num_per_class*task.num_classes, sampler=sampler)

    return loader

