# -*- coding: utf-8 -*-
# @Time    : 2022/12/12 11:16
# @Author  : wth
# @FileName: grad-cam-factor.py
# @Software: PyCharm

import argparse
import os

import cv2
import numpy as np
import torch.cuda
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_pre_process.data_process import DrdatasetMultidataFactor5
from grad_cam_model import GradCAM
from grad_cam_model.cam_utils import show_cam_on_image
from grad_cam_model.get_target_layer import get_nn_target_layer
from network import KeNetMultFactorNew

parser = argparse.ArgumentParser(description='Write control')
parser.add_argument('-s', '--set', type=str, required=True, help='Dataset name')
parser.add_argument('-m', '--model_name', type=str, required=True, help='Model name')
parser.add_argument('-x', '--xlsx_name', type=str, required=True, help='xlsx name')
parser.add_argument('-d', '--dataset_num', type=int, required=True, help='dataset num')
parser.add_argument('-mnp', '--model_name_path', type=str, required=True, help='Model name pth')
parser.add_argument('-b', '--basic_model', type=str, required=True, help='basic pth')
parser.add_argument('-g', '--gpu', type=int, required=True, help='gpu id')
parser.add_argument('-w', '--wam', type=str, required=False, default=False, help="whether to use windows attention")
parser.add_argument('-n', '--win_num', type=int, required=False, default=3, help="The windows number")
parser.add_argument('-el', '--ep_left', type=int, required=False, default=1, help="test begin epoc")
parser.add_argument('-er', '--ep_right', type=int, required=False, default=1, help="test end epoc")
parser.add_argument('-l', '--layer', type=int, required=False, default=1, help="resnet layer")
parser.add_argument('-df', '--dataset', type=str, required=False, default='', help="the number of model")
parser.add_argument('-lb', '--lamb', type=float, required=False, default=1, help="the weight of seg image")
pars = parser.parse_args()

device_ids = [pars.gpu]
test_epoch = range(pars.ep_left, pars.ep_right, 4)[0]
fold = '0'
basic_model = pars.basic_model
basic_net = pars.basic_model
model_name = pars.model_name_path
img_size = 1024
num_class = 2
dataset = 'DKD'
num_thread = 8

if pars.wam == 'True' or pars.wam == 'true':
    windows_attention = True
else:
    windows_attention = False


main_dataset = 'data/cls' + pars.dataset + '/'
Prospective = 'data_test/Prospective/fundus/Prospective/'
Multi_center = 'data_test/Multi_center/'
Non_standard = 'data_test/Non_standard/'
dataset_dic = {
    'main': main_dataset,
    'Prospective': Prospective,
    'Multi_center': Multi_center,
    'Non_standard': Non_standard
}
set_dir = dataset_dic[pars.set]

main_disk = 'data/seg/disk/'
disk_Prospective = 'data_test/Prospective/seg/disk/Prospective/'
disk_Non_standard = 'data_test/Prospective/seg/disk/Non_standard/'
disk_Multi = 'data_test/Multi_center/seg/disk/'

disk_dic = {
    'main': main_disk,
    'Prospective': disk_Prospective,
    'Multi_center': disk_Multi,
    'Non_standard': disk_Non_standard,
}
disk_dir = disk_dic[pars.set]

main_lesion = 'data/seg/lesion/'
lesion_Prospective = 'data_test/Prospective/seg/lesion/Prospective/'
lesion_Non_standard = 'data_test/Prospective/seg/lesion/Non_standard/'
lesion_Multi = 'data_test/Multi_center/seg/lesion/'

lesion_dic = {
    'main': main_lesion,
    'Prospective': lesion_Prospective,
    'Multi_center': lesion_Multi,
    'Non_standard': lesion_Non_standard,
}
lesion_dir = lesion_dic[pars.set]

model_name = pars.model_name_path
model_dir = 'model/' + model_name + '/'

if_after = True
isolate = True

if set_dir == main_dataset:
    isolate = False
else:
    isolate = True
if set_dir == Prospective or set_dir == Non_standard:
    if_after = False
else:
    if_after = True

if set_dir == main_dataset:
    dataset = dataset + '_maindata_test'
    xlsx_path = 'data/risk_factor_5.xlsx'
elif set_dir == Prospective:
    dataset = dataset + '_Prospective'
    xlsx_path = 'data/Prospective_5.xlsx'
elif set_dir == Multi_center:
    dataset = dataset + '_Multi_center'
    xlsx_path = 'data/Multi_center_5.xlsx'
elif set_dir == Non_standard:
    dataset = dataset + '_Non_standard_no'
    xlsx_path = 'data/Prospective_5.xlsx'


def remove_all_file(path):
    if os.path.isdir(path):
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            os.remove(path_file)


class ReshapeTransform:
    def __init__(self, model):
        input_size = model.module.transformer_model.patch_embed.img_size
        patch_size = model.module.transformer_model.patch_embed.patch_size
        self.h = input_size[0] // patch_size[0]
        self.w = input_size[1] // patch_size[1]

    def __call__(self, x):
        # remove cls token and reshape
        # [batch_size, num_tokens, token_dim]
        result = x[:, 1:, :].reshape(x.size(0),
                                     self.h,
                                     self.w,
                                     x.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        # [batch_size, H, W, C] -> [batch, C, H, W]
        result = result.permute(0, 3, 1, 2)
        return result


if __name__ == '__main__':
    if not os.path.isdir('Vis_result/vis_gray'):
        os.makedirs('Vis_result/vis_gray')
    if not os.path.isdir('Vis_result/vis_gray/' + model_name + '_ep' + str(test_epoch)):
        os.makedirs('Vis_result/vis_gray/' + model_name + '_ep' + str(test_epoch))

    if not os.path.isdir('Vis_result/' + model_name + '_ep' + '/' + pars.set + '/' + str(test_epoch)):
        os.makedirs('Vis_result/' + model_name + '_ep' + '/' + pars.set + '/' + str(test_epoch) + '/GT_DN_P_DN')
        os.makedirs('Vis_result/' + model_name + '_ep' + '/' + pars.set + '/' + str(test_epoch) + '/GT_DN_P_NDRD')
        os.makedirs('Vis_result/' + model_name + '_ep' + '/' + pars.set + '/' + str(test_epoch) + '/GT_NDRD_P_DN')
        os.makedirs('Vis_result/' + model_name + '_ep' + '/' + pars.set + '/' + str(test_epoch) + '/GT_NDRD_P_NDRD')
    else:
        remove_all_file('Vis_result/' + model_name + '_ep' + '/' + pars.set + '/' + str(test_epoch) + '/GT_DN_P_DN')
        remove_all_file('Vis_result/' + model_name + '_ep' + '/' + pars.set + '/' + str(test_epoch) + '/GT_DN_P_NDRD')
        remove_all_file('Vis_result/' + model_name + '_ep' + '/' + pars.set + '/' + str(test_epoch) + '/GT_NDRD_P_DN')
        remove_all_file('Vis_result/' + model_name + '_ep' + '/' + pars.set + '/' + str(test_epoch) + '/GT_NDRD_P_NDRD')

    net = KeNetMultFactorNew(classes_num=2, basic_model=basic_model, windows_attention=windows_attention,
                            pretrain=False, windows_num=pars.win_num, initial_method="Uniform", k=0.8,
                            layer_num=pars.layer,lb_weight=pars.lamb).cuda(device_ids[0])
    target_layer = get_nn_target_layer(net, basic_model)
    net = torch.nn.DataParallel(net, device_ids)  # multi-GPUs
    ep_s = str(test_epoch)
    while len(ep_s) < 3:
        ep_s = '0' + ep_s
    a = torch.load(model_dir + 'net_' + ep_s + '.pth', map_location='cpu')
    net.load_state_dict(a)
    # dr_dataset_test = DRDataset(root_img='data/cls/',phase='Test',
    #     img_size=1024, num_class=2, transform=False,fold=fold)
    dr_dataset_test = DrdatasetMultidataFactor5(
        root_img=set_dir,
        root_seg1=disk_dir,
        root_seg2=lesion_dir,
        xlsx_path=xlsx_path,
        phase='Test',
        img_size=img_size, num_class=num_class, transform=False, isolate=isolate, if_after=if_after,
        if_cam=True)
    loader_test = DataLoader(dr_dataset_test, batch_size=1, num_workers=num_thread, shuffle=False)
    test_bar = tqdm(loader_test)

    for packs in test_bar:
        images, seg1, seg2, non_inv_fac, inv_fac, labels = packs[0].cuda(device_ids[0]), packs[1].cuda(
            device_ids[0]), packs[2].cuda(device_ids[0]), packs[3].cuda(device_ids[0]), packs[4].cuda(
            device_ids[0]), packs[5].cuda(device_ids[0])
        if images[0].equal(torch.from_numpy(np.array(-1)).cuda(device_ids[0])):
            continue
        IMG_PATH = packs[6][0]
        net.eval()
        outputs = net(images, seg1, seg2, non_inv_fac, inv_fac)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.cpu().numpy()
        labels = labels.cpu().numpy()

        target_index = 0
        if labels[0] == 0 and predicted[0] == 0:
            img_sub_path = 'GT_DN_P_DN'
            target_index = 0
        if labels[0] == 0 and predicted[0] == 1:
            target_index = 1
            img_sub_path = 'GT_DN_P_NDRD'
        if labels[0] == 1 and predicted[0] == 0:
            target_index = 0
            img_sub_path = 'GT_NDRD_P_DN'
        if labels[0] == 1 and predicted[0] == 1:
            target_index = 1
            img_sub_path = 'GT_NDRD_P_NDRD'

        cam = GradCAM(model=net, target_layers=target_layer, use_cuda=True, device_ids=device_ids)
        grayscale_cam = cam(input_tensor=[images, seg1, seg2, non_inv_fac, inv_fac], target_category=target_index)
        img = images.cpu().numpy()[0]
        img = np.transpose(img, (1, 2, 0))

        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(img.astype(dtype=np.float32),
                                          grayscale_cam,
                                          use_rgb=True)
        dark_visualization = show_cam_on_image(img.astype(dtype=np.float32),
                                               grayscale_cam,
                                               use_rgb=True,
                                               colormap=cv2.COLORMAP_BONE)

        file_name = IMG_PATH.split('/')[-1].split('.')[0]

        plt.imsave('Vis_result/' + model_name + '_ep' + '/' + pars.set + '/' + str(
            test_epoch) + '/' + img_sub_path + '/' + file_name + '_ori.jpg', img)
        # shutil.copy('img_result/seg_alldata/'+IMG_PATH[len('data/cls/DN/after/'):],'Vis_result/' + model_name+'_ep'+str(test_epoch) + '/' + img_sub_path + '/' + IMG_PATH[len('data/cls/DN/after/'):-4] + '_seg.jpg')
        plt.imsave('Vis_result/' + model_name + '_ep' + '/' + pars.set + '/' + str(
            test_epoch) + '/' + img_sub_path + '/' + file_name + '_vis.jpg', grayscale_cam,
                   cmap=plt.get_cmap('gray'))
        plt.imsave('Vis_result/' + model_name + '_ep' + '/' + pars.set + '/' + str(
            test_epoch) + '/' + img_sub_path + '/' + file_name + '_viscombine.jpg', visualization,
                   cmap=plt.get_cmap('jet'))
        plt.imsave('Vis_result/' + model_name + '_ep' + '/' + pars.set + '/' + str(
            test_epoch) + '/' + img_sub_path + '/' + file_name + '_vis_dark.jpg', dark_visualization,)

        plt.imsave('Vis_result/vis_gray/' + model_name + '_ep' + str(test_epoch) + '/' + file_name + '.jpg',
                   grayscale_cam,
                   cmap=plt.get_cmap('gray'))