# -*- coding: utf-8 -*-
import collections

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

import torchvision.transforms as transforms
import os
import glob
from skimage import io
import random
import torch
import platform
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate_err_msg_format, default_collate, np_str_obj_array_pattern
from data_pre_process.xlsx_process import get_id, read_xlsx_5

from torchvision.transforms import *
from PIL import Image
import transform.transforms_group as our_transform
from torchvision.transforms import Compose
import torchvision.transforms.functional as trf

systray = platform.system()

normal_num = 1.0
trans_num = 0


# Data Augment
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


class pretrain_Dataset(Dataset):
    def __init__(self, root_pretrain, phase, transform_r=True):
        self.img_dir = os.path.join(root_pretrain, 'Original_Images', phase, 'img_before')
        self.label_dir = os.path.join(root_pretrain, 'Groundtruths', phase)
        self.img_name_list = os.listdir(self.img_dir)
        self.transform_r = transform_r
        if self.transform_r:
            self.train_transform = train_transform()

    def pil_loader(self, image_path, if_mask=False):
        with open(image_path, 'rb') as f:
            img = Image.open(f)

            if if_mask:
                img = np.array(img)
                img[img > 80] = 255
                img[img <= 80] = 0
                img = Image.fromarray(img.astype('uint8'))
            return img.convert('RGB')

    def __getitem__(self, index):
        img_name = str(self.img_name_list[index])
        txt_name = img_name.split('.')[-2] + '.txt'
        img_dir_index = os.path.join(self.img_dir, img_name)
        label_dir_index = os.path.join(self.label_dir, txt_name)
        if self.transform_r:
            info = self.train_transform([self.pil_loader(img_dir_index, if_mask=False)])[0]
            image = np.array(info)
        else:
            image = io.imread(img_dir_index)

        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.dtype(np.float32))

        f = open(label_dir_index, 'r')
        label = f.readline()
        label = eval(label[0])
        if label > 0:
            label = 1
        else:
            label = 0
        return torch.from_numpy(np.array(image)), torch.from_numpy(np.array(label)).long(), img_name

    def __len__(self):
        return len(self.img_name_list)


class DRDataset(Dataset):
    """
     root：图像存放地址根路径
     augment：是否需要图像增强
    """

    def __init__(self, root_img, phase, img_size, num_class, transform=False, isolate=False, if_after=False):
        # 这个list存放所有图像的地址
        self.phase = phase
        self.image_files = []
        self.root_img = root_img

        if not isolate:
            if self.phase == 'Train' or self.phase == 'Test':
                self.img_list_DN = os.listdir(self.root_img + phase + '/DN/')
                self.img_list_NDRD = os.listdir(self.root_img + phase + '/NDRD/')
                self.image_files = []
                self.image_files = [self.root_img + phase + '/DN/' + name for name in self.img_list_DN] + \
                                   [self.root_img + phase + '/NDRD/' + name for name in self.img_list_NDRD]
        else:
            self.img_list_DN = os.listdir(self.root_img + '/DN/')
            self.img_list_NDRD = os.listdir(self.root_img + '/NDRD/')
            self.image_files = [self.root_img + '/DN/' + name for name in self.img_list_DN] + \
                               [self.root_img + '/NDRD/' + name for name in self.img_list_NDRD]

        self.num_img = len(self.image_files)
        self.img_size = img_size
        self.transform = transform
        self.num_class = num_class
        if transform:
            self.train_transform = train_transform()
        # self.resize_trans = torchvision.transforms.Resize([img_size, img_size])

    def pil_loader(self, image_path, if_mask=False):
        with open(image_path, 'rb') as f:
            img = Image.open(f)

            if if_mask:
                img = np.array(img)
                img[img > 80] = 255
                img[img <= 80] = 0
                img = Image.fromarray(img.astype('uint8'))
            return img.convert('RGB')

    def __getitem__(self, index):

        if self.transform:
            image = self.pil_loader(self.image_files[index], if_mask=False)
            # image = self.resize_trans(image)
            image = self.train_transform([image])[0]
            image = np.array(image)
        else:
            image = self.pil_loader(self.image_files[index], if_mask=False)
            # image = self.resize_trans(image)
            image = np.array(image)

        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.dtype(np.float32))

        if self.image_files[index].split('/')[-2][0] == 'D':
            label = 0
        elif self.image_files[index].split('/')[-2][0] == 'N':
            label = 1

        return torch.from_numpy(np.array(image)), torch.from_numpy(np.array(label)).long(), self.image_files[index]

    def __len__(self):
        return len(self.image_files)


class DrdatasetMultidataFactor5(Dataset):
    """
     root：图像存放地址根路径
     augment：是否需要图像增强
    """

    def __init__(self, root_img, root_seg1, root_seg2, xlsx_path, phase, img_size, num_class, transform=False,
                 isolate=False, if_after=False, if_cam=False):
        # 这个list存放所有图像的地址
        self.phase = phase
        self.image_files = []
        self.seg1_files = []
        self.seg2_files = []
        self.root_img = root_img
        self.root_seg1 = root_seg1
        self.root_seg2 = root_seg2
        self.if_cam = if_cam

        self.ID_Search = {}

        self.ID_data_non_invasive = []
        self.ID_data_invasive = []
        self.ID_Search, self.ID_data_non_invasive, self.ID_data_invasive, self.ID_lesion_type = read_xlsx_5(xlsx_path)

        if not isolate:
            if self.phase == 'Train' or self.phase == 'Test':
                self.img_list_DN = os.listdir(self.root_img + phase + '/DN/')
                self.img_list_NDRD = os.listdir(self.root_img + phase + '/NDRD/')
                self.image_files = [self.root_img + phase + '/DN/' + name for name in self.img_list_DN] + \
                                   [self.root_img + phase + '/NDRD/' + name for name in self.img_list_NDRD]
                for name in self.img_list_DN:
                    if os.path.exists(self.root_seg1 + 'Train' + '/DN/' + name):
                        self.seg1_files.append(self.root_seg1 + 'Train' + '/DN/' + name)
                    else:
                        self.seg1_files.append(self.root_seg1 + 'Test' + '/DN/' + name)
                for name in self.img_list_NDRD:
                    if os.path.exists(self.root_seg1 + 'Train' + '/NDRD/' + name):
                        self.seg1_files.append(self.root_seg1 + 'Train' + '/NDRD/' + name)
                    else:
                        self.seg1_files.append(self.root_seg1 + 'Test' + '/NDRD/' + name)

                for name in self.img_list_DN:
                    if os.path.exists(self.root_seg2 + 'Train' + '/DN/' + name):
                        self.seg2_files.append(self.root_seg2 + 'Train' + '/DN/' + name)
                    else:
                        self.seg2_files.append(self.root_seg2 + 'Test' + '/DN/' + name)
                for name in self.img_list_NDRD:
                    if os.path.exists(self.root_seg2 + 'Train' + '/NDRD/' + name):
                        self.seg2_files.append(self.root_seg2 + 'Train' + '/NDRD/' + name)
                    else:
                        self.seg2_files.append(self.root_seg2 + 'Test' + '/NDRD/' + name)
        else:
            self.img_list_DN = os.listdir(self.root_img + '/DN/')
            self.img_list_NDRD = os.listdir(self.root_img + '/NDRD/')
            self.image_files = [self.root_img + '/DN/' + name for name in self.img_list_DN] + \
                               [self.root_img + '/NDRD/' + name for name in self.img_list_NDRD]
            self.seg1_files = [self.root_seg1 + '/DN/' + name for name in self.img_list_DN] + \
                              [self.root_seg1 + '/NDRD/' + name for name in self.img_list_NDRD]
            self.seg2_files = [self.root_seg2 + '/DN/' + name for name in self.img_list_DN] + \
                              [self.root_seg2 + '/NDRD/' + name for name in self.img_list_NDRD]

        self.num_img = len(self.image_files)
        self.img_size = img_size
        self.transform = transform
        self.num_class = num_class
        if transform:
            self.train_transform = train_transform()

    def pil_loader(self, image_path, if_mask=False):
        with open(image_path, 'rb') as f:
            img = Image.open(f)

            if if_mask:
                img = np.array(img)
                img[img > 80] = 255
                img[img <= 80] = 0
                img = Image.fromarray(img.astype('uint8'))
            return img.convert('RGB')

    def seg_reshape(self, seg, normalize=True):
        if normalize:
            seg = seg / 255.0 / normal_num
        dim_num = seg.ndim
        if dim_num == 3:
            seg = np.mean(seg, 2)
        m, n = seg.shape
        seg = np.reshape(seg, (1, m, n))
        seg = seg.astype(np.dtype(np.float32))
        return seg

    def __getitem__(self, index):

        if self.transform:
            image_pil = self.pil_loader(self.image_files[index], if_mask=False)
            seg1_pil = Image.open(self.seg1_files[index])
            seg2_pil = Image.open(self.seg2_files[index])
            image, seg_1, seg_2 = self.train_transform([image_pil, seg1_pil, seg2_pil])
            image = np.array(image)
            seg_1 = np.array(seg_1)
            seg_2 = np.array(seg_2)
        else:
            image = io.imread(self.image_files[index])
            seg_1 = io.imread(self.seg1_files[index])
            seg_2 = io.imread(self.seg2_files[index])

        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.dtype(np.float32))
        seg_1 = self.seg_reshape(seg_1, normalize=True)
        seg_2 = self.seg_reshape(seg_2, normalize=True)

        if self.image_files[index].split('/')[-2][0] == 'D':
            label = 0
        elif self.image_files[index].split('/')[-2][0] == 'N':
            label = 1

        img_name = self.image_files[index].split('/')
        img_name = img_name[-1]
        id_usr = get_id(img_name)
        id_num = self.ID_Search[id_usr]
        non_invasive = self.ID_data_non_invasive[id_num]
        invasive = self.ID_data_invasive[id_num]
        lesion_type = self.ID_lesion_type[id_num]

        # print(id_usr)
        if min(non_invasive) < 0:
            if self.phase == 'Train':
                if self.if_cam:
                    return torch.from_numpy(np.array(-1)), torch.from_numpy(np.array(-1)), torch.from_numpy(
                        np.array(-1)), torch.from_numpy(np.array(-1)), torch.from_numpy(np.array(-1)), torch.from_numpy(
                        np.array(-1)), torch.from_numpy(np.array(-1))
                else:
                    return None
            elif self.phase == 'Test':
                return torch.from_numpy(np.array(-1)), torch.from_numpy(np.array(-1)), torch.from_numpy(
                    np.array(-1)), torch.from_numpy(np.array(-1)), torch.from_numpy(np.array(-1)), torch.from_numpy(
                    np.array(-1)), torch.from_numpy(np.array(-1))

        return torch.from_numpy(np.array(image)), torch.from_numpy(np.array(seg_1)), torch.from_numpy(np.array(seg_2)), \
            torch.from_numpy(np.array(non_invasive)).to(torch.float32), torch.from_numpy(
            np.array(invasive)).to(torch.float32), torch.from_numpy(
            np.array(label)).long(), self.image_files[index], torch.from_numpy(np.array(lesion_type)).long()

    def __len__(self):
        return len(self.image_files)


######### ########  ########    Seg  ########  ######## ########

def train_transform_seg(degree=30):
    return Compose([
        our_transform.RandomCrop(size=256),
    ])


class SegDataset(Dataset):

    def __init__(self, phase, transform=None):
        self.masks = []
        self.images = []
        seg_label = glob.glob('data/seg/DN/after/*')
        seg_dir_list = os.listdir('data/seg/DN/after/')
        seg_label.sort()
        seg_dir_list.sort()
        for i in range(len(seg_label)):
            if os.path.exists('data/cls/DN/after/' + seg_dir_list[i] + '.jpg'):
                self.masks.append(seg_label[i] + '/lesion_all.jpg')
                self.images.append('data/cls/DN/after/' + seg_dir_list[i] + '.jpg')

        self.phase = phase

        if self.phase == 'Train':
            self.images = self.images[0:int(0.75 * len(self.images))]
            self.masks = self.masks[0:int(0.75 * len(self.masks))]

        elif self.phase == 'Test':
            self.images = self.images[int(0.75 * len(self.images)):]
            self.masks = self.masks[int(0.75 * len(self.masks)):]

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

        return torch.from_numpy(inputs).float(), torch.from_numpy(mask).float(), self.masks[idx]


######### ################# ########     Joint seg and cls  ########  ################# ########

def train_transform_segcls(degree=40):
    return Compose([
        our_transform.RandomVerticalFlip(),
        our_transform.RandomHorizontalFlip(),
        our_transform.RandomRotation(degrees=degree)
    ])


class SegclsData(Dataset):
    """
     root：图像存放地址根路径
     augment：是否需要图像增强
    """

    def __init__(self, mode, root_img, phase, img_size, num_class, transform=False, data_aug=True):
        # 这个list存放所有图像的地址
        self.mode = mode
        self.phase = phase
        self.image_files = []
        self.root_img = root_img
        self.DN_img_list = os.listdir(self.root_img + 'DN/after/')
        self.NDRD_img_list = os.listdir(self.root_img + 'NDRD/after/')
        random.seed(1115)
        random.shuffle(self.DN_img_list, random=random.random)
        random.shuffle(self.NDRD_img_list, random=random.random)
        if self.mode == 'segcls':
            self.seg_dir_list = os.listdir('data/seg/DN/after/') + os.listdir('data/seg/NDRD/after/')

        if self.phase == 'Train':
            for i in range(int(0.75 * len(self.DN_img_list))):
                self.image_files.append(self.root_img + 'DN/after/' + self.DN_img_list[i])
            for i in range(int(0.75 * len(self.NDRD_img_list))):
                self.image_files.append(self.root_img + 'NDRD/after/' + self.NDRD_img_list[i])
        elif self.phase == 'Test':
            for i in range(int(0.75 * len(self.DN_img_list)), int(len(self.DN_img_list))):
                self.image_files.append(self.root_img + 'DN/after/' + self.DN_img_list[i])
            for i in range(int(0.75 * len(self.NDRD_img_list)), int(len(self.NDRD_img_list))):
                self.image_files.append(self.root_img + 'NDRD/after/' + self.NDRD_img_list[i])

        self.num_img = len(self.image_files)
        self.img_size = img_size
        self.transform = transform
        self.num_class = num_class
        if data_aug:
            self.train_transform = train_transform_segcls()

    def pil_loader(self, image_path, if_mask=False):
        with open(image_path, 'rb') as f:
            img = Image.open(f)

            if if_mask:
                img = np.array(img)
                img[img > 80] = 255
                img[img <= 80] = 0
                img = Image.fromarray(img.astype('uint8'))
            return img.convert('RGB')

    def __getitem__(self, index):
        if self.mode == 'cls':
            if self.transform:
                info = [Image.open(self.image_files[index])]
                image = np.array(self.train_transform(info)[0])
            else:
                image = io.imread(self.image_files[index])
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.dtype(np.float32))
            if self.image_files[index][len(self.root_img)] == 'D':
                label = 0
            elif self.image_files[index][len(self.root_img)] == 'N':
                label = 1
            return torch.from_numpy(np.array(image)), torch.from_numpy(np.array(label)).long()


        elif self.mode == 'segcls':
            if_segmap = False
            if self.image_files[index][len(self.root_img)] == 'D':
                if self.image_files[index][len(self.root_img + 'DN/after/'):-4] in self.seg_dir_list:
                    if_segmap = True
                    mask_path = 'data/seg/DN/after/' + self.image_files[index][
                                                       len(self.root_img + 'DN/after/'):-4] + '/lesion_all.jpg'
            if self.image_files[index][len(self.root_img)] == 'N':
                if self.image_files[index][len(self.root_img + 'NDRD/after/'):-4] in self.seg_dir_list:
                    if_segmap = True
                    mask_path = 'data/seg/NDRD/after/' + self.image_files[index][
                                                         len(self.root_img + 'NDRD/after/'):-4] + '/lesion_all.jpg'

            if self.transform:
                info = [self.pil_loader(self.image_files[index], if_mask=False)]
                if if_segmap:
                    info.append(self.pil_loader(mask_path, if_mask=True))
                info = self.train_transform(info)
                image = np.array(info[0])
                if if_segmap:
                    mask = np.array(info[1])
            else:
                image = self.pil_loader(self.image_files[index], if_mask=False)
                image = np.array(image)
                if if_segmap:
                    mask = self.pil_loader(mask_path, if_mask=True)
                    mask = np.array(mask)

            image = np.transpose(image, (2, 0, 1)) / 255.0
            image = image.astype(np.dtype(np.float32))
            if self.image_files[index][len(self.root_img)] == 'D':
                label = 0
            elif self.image_files[index][len(self.root_img)] == 'N':
                label = 1
            if if_segmap:
                mask = np.array(mask)[:, :, 0] / 255.0
                mask = np.array([mask, 1 - mask])
                return torch.from_numpy(np.array(image)), torch.from_numpy(
                    np.array(label)).long(), if_segmap, torch.from_numpy(mask).float()
            else:
                return torch.from_numpy(np.array(image)), torch.from_numpy(np.array(label)).long(), if_segmap

    def __len__(self):
        return len(self.image_files)


def my_default_collate(batch):
    batch_new = []
    for i in range(len(batch)):
        if batch[i] is not None:
            batch_new.append(batch[i])
    batch = batch_new
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel, device=elem.device)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: default_collate([d[key] for d in batch]) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [default_collate(samples) for samples in transposed]  # Backwards compatibility.
        else:
            try:
                return elem_type([default_collate(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


if __name__ == "__main__":
    dr_dataset_train = DRDataset(
        root_img='data/cls/',
        phase='Test',
        img_size=1024, num_class=2, transform=False)
    print(dr_dataset_train.image_files[0])

    # loader_test = DataLoader(dr_dataset_train, batch_size=1, num_workers=3, shuffle=False)
    # print(loader_test[0])
