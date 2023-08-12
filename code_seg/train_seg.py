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

if 1:

    dataset='DKD'
    loss_factor={'Dice':1,'Focal':1}
    optimizer='Adam'
    transform = 'sigmoid'
    num_thread= 8
    train_BATCH_SIZE = 8
    test_BATCH_SIZE = 1
    device_ids = [0]
    EPOCH = 3000
    lr=3*1e-4
    image_size_ori=1024
    image_size_crop=256
    test_threshold=0.5
    model_name = 'seg_final_ablate1'
    num_epochs_decay=100



def main():

    num_lesion=2
    train_dataset = SegDataset(phase='Train',transform=True)
    test_dataset = SegDataset(phase='Test', transform=False)
    print(len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=train_BATCH_SIZE, num_workers=num_thread, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_BATCH_SIZE, num_workers=num_thread, shuffle=False)

    model = Seg_Net(img_ch=3, output_ch=num_lesion).cuda(device_ids[0])
    model = torch.nn.DataParallel(model, device_ids)
    # model.load_state_dict(
    #     torch.load('model/model_U_Net_Cut_Adam_lesion4size_1024crop_256_DDR/model_1451.pth', map_location={'cuda:2': 'cuda:' + str(device_ids[0])}))


    if optimizer == 'SGD':
        g_optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    elif optimizer == 'Adam':
        g_optimizer = optim.Adam(model.parameters(), lr, [0.9, 0.999])
    g_scheduler = lr_scheduler.StepLR(g_optimizer, step_size=200, gamma=0.9)
    Dice_criterion = DiceLoss()
    Focal_criterion = FocalLoss().cuda(device_ids[0])
    writer = SummaryWriter(logdir='runs/runs_' + model_name + '_' + dataset)

    ###########################################
    ##########    train      ##################
    ###########################################

    new_lr=lr
    count = 0
    with open('metrics_' + model_name + '_' + dataset + '.txt', "w+") as f:
        for epoch in range(0, EPOCH):
            step=0
            LOSS=0.
            CE_LOSS=0.
            Dice_LOSS=0.
            Focal_LOSS=0.
            train_bar = tqdm(train_loader)
            if (epoch + 1) > (EPOCH - num_epochs_decay):
                new_lr -= (lr / float(num_epochs_decay))
                for param_group in g_optimizer.param_groups:
                    param_group['lr'] = new_lr
            for inputs, true_masks in train_bar:

                step+=1
                count += 1
                model.train()
                inputs = inputs.cuda(device_ids[0]) #[1,3,256,256]
                true_masks = true_masks.cuda(device_ids[0])#[1,5,256,256]
                true_masks_transpose=true_masks.permute(0, 2, 3, 1)#[1,256,256,5]
                true_masks_pred_flat = true_masks_transpose.reshape(-1, true_masks_transpose.shape[-1])  # [65536,5]
                masks_pred = model(inputs)#[1,5,256,256]
                masks_pred_transpose = masks_pred.permute(0, 2, 3, 1)#[1,256,256,5]
                masks_pred_flat = masks_pred_transpose.reshape(-1, masks_pred_transpose.shape[-1]) #[65536,5]
                true_masks_indices = torch.argmax(true_masks, 1)# [1,256,256]
                ##########################
                #   training             #
                ##########################

                Dice_loss=Dice_criterion(masks_pred_flat, true_masks_pred_flat,transform)
                Focal_loss=Focal_criterion(masks_pred,true_masks_indices,transform)

                Loss= Dice_loss*loss_factor['Dice']+  Focal_loss*loss_factor['Focal']
                g_optimizer.zero_grad()
                Loss.backward()
                g_optimizer.step()
                LOSS+=Loss.item()
                Dice_LOSS+=Dice_loss.item()*loss_factor['Dice']
                Focal_LOSS+=Focal_loss.item()*loss_factor['Focal']

                train_bar.set_description(desc='[%d/%d] Loss: %.4f | Dice_loss: %.4f | Focal_loss: %.4f' % (
                    epoch, EPOCH,
                    LOSS/step, Dice_LOSS/step, Focal_LOSS/step
                ))

                """------------------tensorboard TRAIN--------------"""
                if count % 100 == 0:
                    writer.add_scalar('scalar/Loss', Loss.item(), count)
                    writer.add_scalar('scalar/Dice_loss', Dice_loss.item() *loss_factor['Dice'], count)
                    writer.add_scalar('scalar/Focal_loss', Focal_loss.item() *loss_factor['Focal'], count)

            # """------------------Test--------------"""

            if epoch % 50 == 0:
                print("Waiting Test!")
                with torch.no_grad():
                    batch_sizes=0
                    metrics={'Acc':[],'AUC_ROC':[],'AUC_PR':[]}
                    test_bar = tqdm(test_loader)
                    for inputs, true_masks in test_bar:
                        model.eval()
                        inputs = inputs.cuda(device_ids[0])  # [1,3,1024,1024]
                        true_masks = true_masks.cuda(device_ids[0]) #  [1,5,1024,1024]
                        GT=true_masks.cpu().numpy()[0][0].reshape(-1).astype(int) # [1,1024,1024]

                        batch_sizes = batch_sizes+test_BATCH_SIZE
                        masks_pred = model(inputs)

                        if transform == 'softmax':
                            masks_pred = F.softmax(masks_pred, dim=1)  # [1,5,1024,1024]
                        else:
                            masks_pred = torch.sigmoid(masks_pred)  # [1,5,1024,1024]
                        result = masks_pred.cpu().numpy()[0] # [5,1024,1024]
                        lesion_pred_scores=result[0].reshape(-1)
                        pred_binary =np.array(result[0].reshape(-1) >test_threshold).astype(int)
                        metrics['Acc'].append(accuracy_score(y_true= GT,y_pred=pred_binary))
                        if np.sum(GT) != 0:
                            metrics['AUC_ROC'].append(roc_auc_score(y_true= GT, y_score=lesion_pred_scores))

                        if np.sum(GT) != 0:
                            metrics['AUC_PR'].append(average_precision_score(y_true= GT, y_score=lesion_pred_scores))


                Acc=np.mean(np.array(metrics['Acc']))
                AUC_ROC = np.mean(np.array(metrics['AUC_ROC']))
                AUC_PR = np.mean(np.array(metrics['AUC_PR']))
                print('Testset  PR=：%.3f%%'
                    % (AUC_PR))
                f.write("EPOCH=%03d | Acc=%.4f  AUC_ROC=%.4f| AUC_PR=%.4f"
                        % (epoch + 1, Acc, AUC_ROC, AUC_PR))
                f.write('\n')
                f.flush()
                writer.add_scalar('scalar/AUC_ROC_MA', AUC_ROC, epoch)
                writer.add_scalar('scalar/AUC_PR', AUC_PR, epoch)


                torch.save(model.state_dict(),'model/model_' + model_name + '_' + dataset + '/model_' + str(epoch + 1) + '.pth')
                print('Test: AUC_PR=%.4f ' % (AUC_PR))

            g_scheduler.step()
        writer.close()


def remove_all_file(path):
    if os.path.isdir(path):
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            os.remove(path_file)


def test():

    num_lesion=2
    test_dataset = SegDataset(phase='Test', transform=False)
    test_loader = DataLoader(test_dataset, batch_size=test_BATCH_SIZE, num_workers=num_thread, shuffle=False)

    model = Seg_Net(img_ch=3, output_ch=num_lesion).cuda(device_ids[0])
    model = torch.nn.DataParallel(model, device_ids)
    model.load_state_dict(
        torch.load('model/model_2951.pth', map_location={'cuda:0': 'cuda:' + str(device_ids[0])}))


    with open('metrics_' + model_name + '_' + dataset + '.txt', "w+") as f:
        print("Waiting Test!")
        with torch.no_grad():
            batch_sizes=0
            test_bar = tqdm(test_loader)
            for inputs, true_masks, IMG_PATH in test_bar:
                IMG_PATH = IMG_PATH[0].split('/')[-2]
                model.eval()
                inputs = inputs.cuda(device_ids[0])
                true_masks = true_masks.cuda(device_ids[0])
                GT=true_masks.cpu().numpy()[0][0].reshape(-1).astype(int)

                batch_sizes = batch_sizes+test_BATCH_SIZE
                masks_pred = model(inputs)

                if transform == 'softmax':
                    masks_pred = F.softmax(masks_pred, dim=1)
                else:
                    masks_pred = torch.sigmoid(masks_pred)
                result = masks_pred.cpu().numpy()[0] # [5,1024,1024]
                lesion_pred_scores=result[0].reshape(-1)
                pred_binary =np.array(result[0].reshape(-1) >test_threshold).astype(int)
                plt.imsave('img_result/' + IMG_PATH + '.jpg', np.array(result[0] >test_threshold).astype(int),
                           cmap=plt.get_cmap('gray'))

                if np.sum(GT) != 0:
                    f.write("%s "%(IMG_PATH.split('/')[-1]))
                    f.write("%.3f "%(dice(pred_binary, GT)))
                    f.write("%.3f " %(jaccard(pred_binary, GT)))
                    f.write("%.3f " %(roc_auc_score(y_true= GT, y_score=lesion_pred_scores)))
                    f.write("%.3f " % (precision(pred_binary, GT)))
                    f.write('\n')
                    f.flush()
        f.close()

if __name__ == "__main__":

    init_seed = 915
    np.random.seed(init_seed)
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    import platform

    sysstr = platform.system()
    if (sysstr == "Linux"):
        if not os.path.isdir('model/model_'+model_name+'_'+dataset):
            os.makedirs('model/model_'+model_name+'_'+dataset)
        if not os.path.isdir('runs/runs_' + model_name + '_' + dataset):
            os.makedirs('runs/runs_' + model_name + '_' + dataset)
        # else:
        #     remove_all_file('model/model_'+model_name+'_'+dataset)


        # if os.path.isdir('runs/runs_'+model_name+'_'+dataset):
        #     remove_all_file('runs/runs_' + model_name + '_' + dataset)

    main()
    # test()
