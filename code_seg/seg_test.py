# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import numpy as np
import math
import os
import glob
import random
from network import Seg_Net
from data_process import SegDataset
from torchvision import datasets, models, transforms
from transform.transforms_group import *
from torch.utils.data import DataLoader, Dataset
from torch.optim import lr_scheduler
from loss import FocalLoss,DiceLoss
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,accuracy_score,average_precision_score
from sklearn.metrics import auc
from tqdm import tqdm
from metrics_seg import *
from PIL import Image
from metrics_seg  import  dice
init_seed = 915
np.random.seed(init_seed)
torch.manual_seed(init_seed)
torch.cuda.manual_seed_all(init_seed)


if 1:

    dataset='DKD'
    num_thread= 8
    test_BATCH_SIZE = 1
    device_ids = [0]
    image_size_ori=1024
    image_size_crop=256
    test_threshold=0.5
    model_name = 'model_2951'

color_map={ 'soft exudates':[139 / 255, 20 / 255, 8 / 255],'haemorrhages':[188 / 255, 189 / 255, 34 / 255],
                  'hard exudates':[52 / 255, 193 / 255, 52 / 255],
                  'drusen':[150 / 255, 150 / 255, 190 / 255], 'neovascularization':[139 / 255, 101 / 255, 8 / 255],
                  'microaneurysms':[68 / 255, 114 / 255, 236 / 255],
                  'pigmentation':[100 / 255, 114 / 255, 196 / 255], 'retinitis pigmentosa atrophy':[214 / 255 + 0.1, 39 / 255 + 0.2, 40 / 255 + 0.2],
          'Preretinal macular membrane': [52 / 255, 163 / 255, 152 / 255],
            'fibre-added membrance': [236 / 255, 193 / 255, 52 / 255],
            'discus opticus membrane':[52 / 255, 236 / 255, 52 / 255],
            'old preretinal hemorrhage':[52 / 255, 193 / 255, 236 / 255],
'retinal detachment':[236 / 255, 20 / 255, 8 / 255],
'microvascular proliferation':[139 / 255, 236 / 255, 8 / 255],
'epiretinal membrance':[139 / 255, 20 / 255, 236 / 255],
'edama':[236 / 255, 114 / 255, 236 / 255],
'vitreum floats':[236 / 255, 253 / 255, 236 / 255],
'laser scars':[236 / 255, 253 / 255, 8 / 255],
'lasser scars':[236 / 255, 253 / 255, 8 / 255],
'membrance':[139 / 255, 253 / 255, 8 / 255]
            }

for key in color_map:
    lesion_colormap=np.ones(shape=(1024,1024,3), dtype=float)
    lesion_colormap=lesion_colormap*color_map[key]
    plt.imsave('img_result/reference_lesion_color/' + key+ '.jpg',
               lesion_colormap)


def remove_all_file(path):
    if os.path.isdir(path):
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            os.remove(path_file)
if not os.path.isdir('img_result/' + model_name):
    os.makedirs('img_result/' + model_name)
else:
    remove_all_file('img_result/' + model_name)

num_lesion=2
test_dataset = SegDataset(phase='Test', transform=False)
test_loader = DataLoader(test_dataset, batch_size=test_BATCH_SIZE, num_workers=num_thread, shuffle=False)

model = Seg_Net(img_ch=3, output_ch=num_lesion).cuda(device_ids[0])
model = torch.nn.DataParallel(model, device_ids)
model.load_state_dict(
    torch.load('model/'+model_name+'.pth', map_location={'cuda:3': 'cuda:' + str(device_ids[0])}))

print("Waiting Test!")
with torch.no_grad():
    batch_sizes=0
    metrics={'Acc':[],'AUC_ROC':[],'AUC_ROC_BG':[],'AUC_PR':[],'AUC_PR_BG':[],'dice':[],'dice_BG':[]}
    test_bar = tqdm(test_loader)
    for inputs, true_masks , MASK_PATH, IMG_PATH in test_bar:
        mask_path_root=MASK_PATH[0][:-14]
        MASK_NAME = MASK_PATH[0].split('/')[-2]
        IMG_PATH = IMG_PATH[0]
        model.eval()
        inputs = inputs.cuda(device_ids[0])  # [1,3,1024,1024]
        true_masks = true_masks.cuda(device_ids[0]) #  [1,5,1024,1024]
        GT=true_masks.cpu().numpy()[0][0].reshape(-1).astype(int) # [1,1024,1024]

        batch_sizes = batch_sizes+test_BATCH_SIZE
        masks_pred = model(inputs)

        masks_pred = torch.sigmoid(masks_pred)  # [1,5,1024,1024]
        result = masks_pred.cpu().numpy()[0] # [5,1024,1024]
        lesion_pred_scores=result[0].reshape(-1)
        pred_binary =np.array(result[0].reshape(-1) >test_threshold).astype(int)

        # metrics['Acc'].append(accuracy_score(y_true= GT,y_pred=pred_binary))
        # if np.sum(GT) != 0:
        #     metrics['AUC_ROC'].append(roc_auc_score(y_true= GT, y_score=lesion_pred_scores))
        # if np.sum(true_masks.cpu().numpy()[0][1].reshape(-1).astype(int)) != 0:
        #     metrics['AUC_ROC_BG'].append(roc_auc_score(y_true=true_masks.cpu().numpy()[0][1].reshape(-1).astype(int), y_score=result[1].reshape(-1)))
        #
        # if np.sum(GT) != 0:
        #     metrics['AUC_PR'].append(average_precision_score(y_true= GT, y_score=lesion_pred_scores))
        # if np.sum(true_masks.cpu().numpy()[0][1].reshape(-1).astype(int)) != 0:
        #     metrics['AUC_PR_BG'].append(average_precision_score(y_true=true_masks.cpu().numpy()[0][1].reshape(-1).astype(int), y_score=result[1].reshape(-1)))
        #
        # if np.sum(GT) != 0:
        #     metrics['dice'].append(dice(np.expand_dims(pred_binary, axis=0), np.expand_dims(GT, axis=0)))
        # if np.sum(true_masks.cpu().numpy()[0][1].reshape(-1).astype(int)) != 0:
        #     metrics['dice_BG'].append(dice(np.expand_dims(np.array(result[1].reshape(-1) >test_threshold).astype(int), axis=0),np.expand_dims(true_masks.cpu().numpy()[0][1].reshape(-1).astype(int), axis=0)))



        GTmask_img=np.array(Image.open(MASK_PATH[0]).convert('RGB'))/ 255 ## 0-1
        ori_img=np.array(Image.open(IMG_PATH).convert('RGB'))/ 255
        Predmask_img0=np.expand_dims(np.array(result[0]), axis=2)
        Predmask_img=np.concatenate((Predmask_img0, Predmask_img0, Predmask_img0), axis=2)
        Predmask_img_max = np.max(Predmask_img)
        Predmask_img_min = np.min(Predmask_img)
        if not Predmask_img_min:
            Predmask_img=(Predmask_img-Predmask_img_min)/Predmask_img_max
        Predmask_img[Predmask_img < 0.5] = 0
        Predmask_img[Predmask_img >= 0.5] = 1

        gt_lesions=os.listdir(mask_path_root)
        count=0
        for k in range(len(gt_lesions)):
            if not gt_lesions[k]=='lesion_all.jpg':
                color = color_map[gt_lesions[k][:-4]]
                if count==0:
                    img_lesion = np.array(Image.open(mask_path_root+gt_lesions[k]).convert('RGB'))/255*color
                    count+=1
                else:
                    # plt.imshow(img_lesion)
                    # plt.show()
                    new_img_lesion=np.array(Image.open(mask_path_root + gt_lesions[k]).convert('RGB')) / 255 * color
                    logic_new_img_lesion=(new_img_lesion>0)* 1
                    logic_img_lesion = (img_lesion > 0) * 1
                    new_img_lesion[logic_img_lesion*logic_new_img_lesion>0]=0
                    img_lesion+=new_img_lesion
        # plt.imshow(img_lesion)
        # plt.show()
        # plt.imsave('img_result/' + model_name + '/' + MASK_NAME + '_ori.jpg',
        #            ori_img)
        # plt.imsave('img_result/' + model_name + '/' + MASK_NAME + '_gtbinary.jpg',
        #            GTmask_img[...,0],cmap=plt.get_cmap('viridis'))
        plt.imsave('img_result/'+ model_name+ '/' + MASK_NAME + '_pred_base.jpg', Predmask_img[...,0],
                   cmap=plt.get_cmap('viridis'))
        # plt.imsave('img_result/' + model_name + '/' + MASK_NAME + '_gtmultiple.jpg',
        #            img_lesion)
        a=1


Acc=np.mean(np.array(metrics['Acc']))
AUC_ROC = np.mean(np.array(metrics['AUC_ROC']))
AUC_ROC_BG = np.mean(np.array(metrics['AUC_ROC_BG']))
AUC_PR = np.mean(np.array(metrics['AUC_PR']))
AUC_PR_BG = np.mean(np.array(metrics['AUC_PR_BG']))
dice = np.mean(np.array(metrics['dice']))
dice_BG = np.mean(np.array(metrics['dice_BG']))
print('Testset  Acc=：%.3f%%' % (Acc))
print('Testset  AUC_ROC=：%.3f%%' % ((AUC_ROC+AUC_ROC_BG)/2))
print('Testset  AUC_PR=：%.3f%%' % ((AUC_PR+AUC_PR_BG)/2))
print('Testset  dice=：%.3f%%' % ((dice+dice_BG)/2))

















