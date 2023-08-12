# -*- coding: utf-8 -*-
import collections

import torchvision

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

import torchvision.transforms as transforms
import os
import glob
from skimage import io, transform
import random
import torch
import platform
import numpy as np
# from torch._six import string_classes
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
import cv2
from torch.utils.data._utils.collate import default_collate_err_msg_format, default_collate, np_str_obj_array_pattern
from torch.utils.data.dataset import T_co


sysstr = platform.system()
from sklearn.neighbors import NearestNeighbors
from torchvision.transforms import *
import torch.utils.data as data
from PIL import Image
import transform.transforms_group as our_transform
from torchvision.transforms import Compose
import torchvision.transforms.functional as trf

normal_num = 1.0
trans_num = 0


def train_transform(degree=180):
    number = random.random()
    if number > trans_num:
        return Compose([
            # our_transform.AddGaussianNoise(mean=0.0, variance=0.1, amplitude=0.1),
            our_transform.RandomVerticalFlip(),

            our_transform.RandomHorizontalFlip(),
            our_transform.RandomRotation(degrees=degree),
            # our_transform.RandomAffine(degrees=0, translate=(0, 0), shear=20),
            our_transform.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05),
        ])
    else:
        return Compose([
            # our_transform.AddGaussianNoise(mean=0.0, variance=0.1, amplitude=0.1),
            our_transform.RandomVerticalFlip(),
            our_transform.RandomHorizontalFlip(),
            # our_transform.RandomRotation(degrees=degree),
            # our_transform.RandomAffine(degrees=0, translate=(0, 0), shear=20),
            # our_transform.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05),
        ])


def train_transform_cat(image, seg1, seg2, degree=30):
    image = my_colorjitter(image)
    image, seg1, seg2 = my_flip(image, seg1, seg2)
    image, seg1, seg2 = my_rotate(image, seg1, seg2, degree=degree)
    return image, seg1, seg2


def my_rotate(image, seg1, seg2, degree):
    angle = transforms.RandomRotation.get_params([-1.0 * degree, degree])
    image = image.rotate(angle)
    seg1 = seg1.rotate(angle)
    seg2 = seg2.rotate(angle)
    return image, seg1, seg2


def my_flip(image, seg1, seg2):
    if random.random() > 0.5:
        image = trf.hflip(image)
        seg1 = trf.hflip(seg1)
        seg2 = trf.hflip(seg2)

    if random.random() > 0.5:
        image = trf.vflip(image)
        seg1 = trf.vflip(seg1)
        seg2 = trf.vflip(seg2)
    return image, seg1, seg2


def my_colorjitter(image):
    trans = ColorJitter()
    image = trans(image)
    return image

def train_transform_seg(degree=30):
    return Compose([
        our_transform.RandomCrop(size=256),
    ])



class SegDataset(Dataset):

    def __init__(self, phase, transform=None):
        self.masks_DN_clstrain = []
        self.images_DN_clstrain = []
        self.masks_DN_clstest = []
        self.images_DN_clstest = []
        self.masks_NDRD_clstrain = []
        self.images_NDRD_clstrain = []
        self.masks_NDRD_clstest = []
        self.images_NDRD_clstest = []
        # cls_path=r'/mnt/ssd4/wengtaohan/DKD/data/cls_4/'
        # seg_label_DN = glob.glob('../DKD/data/seg/DN/after/*')
        # seg_dir_list_DN = os.listdir('../DKD/data/seg/DN/after/')
        # seg_label_NDRD = glob.glob('../DKD/data/seg/NDRD/after/*')
        # seg_dir_list_NDRD = os.listdir('../DKD/data/seg/NDRD/after/')

        cls_path = r'./'
        seg_label_DN = glob.glob('./seg/DN/after/*')
        seg_dir_list_DN = os.listdir('./seg/DN/after/')
        seg_label_NDRD = glob.glob('./seg/NDRD/after/*')
        seg_dir_list_NDRD = os.listdir('./seg/NDRD/after/')

        ### all image
        count =0
        seg_label_DN.sort()
        seg_dir_list_DN.sort()
        for i in range(len(seg_label_DN)):
            if os.path.exists(cls_path + 'Train/DN/' + seg_dir_list_DN[i] + '.jpg'):
                count += 1
                self.masks_DN_clstrain.append(seg_label_DN[i] + '/lesion_all.jpg')
                self.images_DN_clstrain.append(cls_path + 'Train/DN/' + seg_dir_list_DN[i] + '.jpg')
            if os.path.exists(cls_path + 'Test/DN/' + seg_dir_list_DN[i] + '.jpg'):
                count += 1
                self.masks_DN_clstest.append(seg_label_DN[i] + '/lesion_all.jpg')
                self.images_DN_clstest.append(cls_path + 'Test/DN/' + seg_dir_list_DN[i] + '.jpg')
            if os.path.exists(cls_path + 'Train/NDRD/' + seg_dir_list_DN[i] + '.jpg'):
                count += 1
                self.masks_DN_clstrain.append(seg_label_DN[i] + '/lesion_all.jpg')
                self.images_DN_clstrain.append(cls_path + 'Train/NDRD/' + seg_dir_list_DN[i] + '.jpg')
            if os.path.exists(cls_path + 'Test/NDRD/' + seg_dir_list_DN[i] + '.jpg'):
                count += 1
                self.masks_DN_clstest.append(seg_label_DN[i] + '/lesion_all.jpg')
                self.images_DN_clstest.append(cls_path + 'Test/NDRD/' + seg_dir_list_DN[i] + '.jpg')



        seg_label_NDRD.sort()
        seg_dir_list_NDRD.sort()
        for i in range(len(seg_label_NDRD)):
            if os.path.exists(cls_path + 'Train/NDRD/' + seg_dir_list_NDRD[i] + '.jpg'):
                self.masks_NDRD_clstrain.append(seg_label_NDRD[i] + '/lesion_all.jpg')
                self.images_NDRD_clstrain.append(cls_path + 'Train/NDRD/' + seg_dir_list_NDRD[i] + '.jpg')
            if os.path.exists(cls_path + 'Test/NDRD/' + seg_dir_list_NDRD[i] + '.jpg'):
                self.masks_NDRD_clstest.append(seg_label_NDRD[i] + '/lesion_all.jpg')
                self.images_NDRD_clstest.append(cls_path + 'Test/NDRD/' + seg_dir_list_NDRD[i] + '.jpg')
            if os.path.exists(cls_path + 'Train/DN/' + seg_dir_list_NDRD[i] + '.jpg'):
                self.masks_NDRD_clstrain.append(seg_label_NDRD[i] + '/lesion_all.jpg')
                self.images_NDRD_clstrain.append(cls_path + 'Train/DN/' + seg_dir_list_NDRD[i] + '.jpg')
            if os.path.exists(cls_path + 'Test/DN/' + seg_dir_list_NDRD[i] + '.jpg'):
                self.masks_NDRD_clstest.append(seg_label_NDRD[i] + '/lesion_all.jpg')
                self.images_NDRD_clstest.append(cls_path + 'Test/DN/' + seg_dir_list_NDRD[i] + '.jpg')





        self.phase = phase

        if self.phase == 'Train':
            self.images = self.images_DN_clstrain+self.images_NDRD_clstrain
            self.masks = self.masks_DN_clstrain+self.masks_NDRD_clstrain

        elif self.phase == 'Test':
            self.images = self.images_DN_clstest + self.images_NDRD_clstest
            self.masks = self.masks_DN_clstest + self.masks_NDRD_clstest

        self.transform = transform
        self.transform_func = train_transform_seg()
        self.image_size_ori = 1024

    def __len__(self):
        return len(self.images)

    def pil_loader(self, image_path, if_mask=False):
        with open(image_path, 'rb') as f:
            img = Image.open(f)

            if if_mask:
                img = np.array(img)
                img[img > 80] = 255
                img[img <= 80] = 0
                img = Image.fromarray(img.astype('uint8'))
            return img.convert('RGB')

    def __getitem__(self, idx):

        info = [self.pil_loader(self.images[idx], if_mask=False)]
        info.append(self.pil_loader(self.masks[idx], if_mask=True))

        if self.transform:
            info = self.transform_func(info)

        inputs = np.array(info[0])
        if inputs.shape[2] == 3:
            inputs = np.transpose(np.array(info[0]), (2, 0, 1))
            inputs = inputs / 255.

        mask = np.array(info[1])[:, :, 0] / 255.0
        mask = np.array([mask, 1 - mask])

        return torch.from_numpy(inputs).float(), torch.from_numpy(mask).float(), self.masks[idx],self.images[idx]


######### ################# ########     Joint seg and cls  ########  ################# ########

def train_transform_segcls(degree=40):
    return Compose([
        our_transform.RandomVerticalFlip(),
        our_transform.RandomHorizontalFlip(),
        our_transform.RandomRotation(degrees=degree)
    ])



if __name__ == "__main__":
    a=1
    train_dataset = SegDataset(phase='Train',transform=True)
    a=train_dataset._getitem__(1)
    #     root_img='data/cls/',
    #     phase='Test',
    #     img_size=1024, num_class=2, transform=False, fold=4)
    # print(dr_dataset_train.image_files[0])

    # loader_test = DataLoader(dr_dataset_train, batch_size=1, num_workers=3, shuffle=False)
    # print(loader_test[0])
