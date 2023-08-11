import torch.nn as nn
import torch.optim as optim
from data_pre_process.data_process import my_default_collate, DRDataset_Multidata_factor_5
import numpy as np
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import os
import torch.cuda
from tqdm import tqdm
import argparse
from network import KeNetMultFactorNew

# python train_fusion.py -b m3 -g 0 -pre False -f 0 -w False -n 4 -lr 1 -bc 8 -e 150 -l_num 1 -d _4

parser = argparse.ArgumentParser(description='basic net')
parser.add_argument('-b', '--basic_net', type=str, required=True, help='basic_net')
parser.add_argument('-g', '--gpu', type=int, required=True, help='gpu id')
parser.add_argument('-pre', '--pretrain', type=str, required=True, help='pretrain')
parser.add_argument('-f', '--fold', type=str, required=True, help='batch size')
parser.add_argument('-w', '--wam', type=str, required=False, default=False, help="whether to use windows attention")
parser.add_argument('-n', '--win_num', type=int, required=False, default=3, help="The windows number")
parser.add_argument('-lr', '--LR', type=int, required=False, default=1, help="Learning rate")
parser.add_argument('-bc', '--batch', type=int, required=False, default=4, help="batch_size")
parser.add_argument('-e', '--epoc', type=int, required=False, default=800, help="epoc_num")
parser.add_argument('-l_num', '--layer_num', type=int, required=False, default=1, help="the num of resnet wam layer")
parser.add_argument('-ld', '--load_model', type=int, required=False, default='-1', help="the number of model")
parser.add_argument('-d', '--dataset', type=str, required=False, default='', help="dataset describe")
args = parser.parse_args()

EPOCH = args.epoc
num_epochs_decay = 600
img_size = 1024
num_class = 2
dataset = 'DKD'
num_thread = 8
device_ids = [args.gpu]
basic_model = args.basic_net  # inception  densenet resnet
fold = args.fold
pretrain = True if args.pretrain == 'True' else False

from config.cls_v3 import *

if args.wam == 'True' or args.wam == 'true':
    windows_attention = True
else:
    windows_attention = False

net = KeNetMultFactorNew(classes_num=num_class, basic_model=basic_model, windows_attention=windows_attention,
                         pretrain=pretrain, windows_num=args.win_num, initial_method="Uniform", k=0.8,
                         layer_num=args.layer_num).cuda(device_ids[0])
net = torch.nn.DataParallel(net, device_ids)

# load the pretrain model
if pretrain:
    pre_model_dir = args.pre_path
    save_model = torch.load(pre_model_dir)
    model_dict = net.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    net.load_state_dict(model_dict)
# sf:small factor
model_name = '7.30_contrast_' + basic_model + 'foldn' + args.dataset + '_win_num' + str(args.win_num) + '_lr' + str(
    args.LR) + '_ep' + str(
    args.epoc) + 'bc_' + str(args.batch) + 'lnum_' + str(args.layer_num)
if args.wam == "True" or args.wam == "true":
    model_name = 'begin_wam_' + model_name
if not pretrain:
    model_name = model_name + '_Nopretrain'

epoc_begin = 0
if args.load_model > 0:
    net.load_state_dict(
        torch.load('%s/net_%03d.pth' % ('model/model_' + model_name + '_' + dataset, args.load_model)))
    epoc_begin = args.load_model
train_BATCH_SIZE = args.batch
test_BATCH_SIZE = 1


def main():
    criterion = nn.CrossEntropyLoss(
        weight=torch.from_numpy(np.array([weight[0], weight[1]])).float().cuda(device_ids[0]))
    optimizer = optim.Adam(net.parameters(), lr=LR * args.LR, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    dr_dataset_train = DRDataset_Multidata_factor_5(root_img='data/' + 'cls' + args.dataset + '/',
                                                    root_seg1='data/seg/disk/',
                                                    root_seg2='data/seg/lesion/',
                                                    xlsx_path='data/risk_factor_5.xlsx',
                                                    phase='Train',
                                                    img_size=img_size, num_class=num_class, transform=True, fold=fold,
                                                    if_after=True)
    dr_dataset_test = DRDataset_Multidata_factor_5(
        root_img='data/' + 'cls' + args.dataset + '/',
        root_seg1='data/seg/disk/',
        root_seg2='data/seg/lesion/',
        xlsx_path='data/risk_factor_5.xlsx',
        phase='Test',
        img_size=img_size, num_class=num_class, transform=False, fold=fold, if_after=True)

    loader_train = DataLoader(dr_dataset_train, batch_size=train_BATCH_SIZE, num_workers=num_thread, shuffle=True,
                              collate_fn=my_default_collate, drop_last=True)
    loader_test = DataLoader(dr_dataset_test, batch_size=test_BATCH_SIZE, num_workers=num_thread, shuffle=False)
    writer = SummaryWriter(logdir='runs/runs_' + model_name + '_' + dataset)
    count_all = 0
    new_lr = LR * args.LR
    with open('acc/acc_' + model_name + '_' + dataset + '.txt', "w+") as f:
        for epoch in range(epoc_begin, EPOCH):
            # if (epoch + 1) > (EPOCH - num_epochs_decay):
            #     new_lr -= (LR / float(num_epochs_decay))
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = new_lr
            #     print('Decay learning rate to lr: {}.'.format(new_lr))

            running_results = {'acc': 0, 'acc_loss': 0}
            print('Decay learning rate to lr: {}.'.format(optimizer.param_groups[0]['lr']))
            train_bar = tqdm(loader_train)
            count = 0
            """--------------------------------------Train---------------------------------------"""
            for packs in train_bar:
                count += 1
                count_all += 1
                net.train()
                inputs, seg1, seg2, non_inv_fac, inv_fac, labels, labels_split = packs[0].cuda(device_ids[0]), packs[
                    1].cuda(
                    device_ids[0]), packs[2].cuda(device_ids[0]), packs[3].cuda(device_ids[0]), packs[4].cuda(
                    device_ids[0]), packs[5].cuda(device_ids[0]), packs[7].cuda(device_ids[0])
                # labels_split =
                optimizer.zero_grad()

                outputs = net(inputs, seg1, seg2, non_inv_fac, inv_fac)
                loss_ce = criterion(outputs, labels)  # vanilla softmax loss

                _, predicted = torch.max(outputs.data, 1)
                loss = loss_ce

                loss.backward()
                optimizer.step()

                total = labels.size(0)
                correct = predicted.eq(labels.data).cpu().sum()

                running_results['acc'] += 100. * correct / total
                running_results['acc_loss'] += loss.item()

                train_bar.set_description(
                    desc=model_name + ' [%d/%d] acc_loss: %.4f  ' % (
                        epoch, EPOCH,
                        running_results['acc_loss'] / count
                    ))

                """------------------tensorboard test--------------"""
                if count % 4 == 0:
                    writer.add_scalar('scalar/train_loss_per_iter', loss.item(), count_all)
                    writer.add_scalar('scalar/acc_batchwise', (100. * correct / total), count_all)

            """------------------Test--------------"""

            if epoch % 4 == 0:
                test_bar = tqdm(loader_test)
                print("Waiting Test!")
                with torch.no_grad():
                    correct_all = 0
                    total_all = 0
                    tp = 0
                    tn = 0
                    fp = 0
                    fn = 0
                    for packs in test_bar:
                        net.eval()
                        images, seg1, seg2, non_inv_fac, inv_fac, labels = packs[0].cuda(device_ids[0]), packs[1].cuda(
                            device_ids[0]), packs[2].cuda(device_ids[0]), packs[3].cuda(device_ids[0]), packs[4].cuda(
                            device_ids[0]), packs[5].cuda(device_ids[0])
                        if images[0].equal(torch.from_numpy(np.array(-1)).cuda(device_ids[0])):
                            continue
                        outputs = net(images, seg1, seg2, non_inv_fac, inv_fac)
                        _, predicted = torch.max(outputs.data, 1)
                        total_all += labels.size(0)
                        correct_all += (predicted == labels).sum()
                        labels = labels.cpu().numpy()
                        predicted = predicted.cpu().numpy()

                        for i_test in range(test_BATCH_SIZE):
                            if labels[i_test] == 1 and predicted[i_test] == 1:
                                tp += 1
                            if labels[i_test] == 1 and predicted[i_test] == 0:
                                fn += 1
                            if labels[i_test] == 0 and predicted[i_test] == 1:
                                fp += 1
                            if labels[i_test] == 0 and predicted[i_test] == 0:
                                tn += 1

                    Acc = (tp + tn) / (tp + tn + fp + fn)
                    Sen = (tp) / (tp + fn)
                    Spec = (tn) / (tn + fp)
                    print('Testset Acc=：%.1f%% | Sen=：%.1f%% | Spec=：%.1f%% ' % (Acc * 100, Sen * 100, Spec * 100))

                    torch.save(net.state_dict(),
                               '%s/net_%03d.pth' % ('model/model_' + model_name + '_' + dataset, epoch + 1))
                    f.write("EPOCH=%03d | Acc=：%.1f%% | Sen=：%.1f%% | Spec=：%.1f%% "
                            % (epoch + 1, Acc * 100, Sen * 100, Spec * 100))
                    f.write('\n')
                    f.flush()

                    writer.add_scalar('scalar/test_Acc', Acc, epoch)
                    writer.add_scalar('scalar/test_Sen', Sen, epoch)
                    writer.add_scalar('scalar/test_Spec', Spec, epoch)

            scheduler.step()
        writer.close()


def remove_all_file(path):
    if os.path.isdir(path):
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            os.remove(path_file)


if __name__ == "__main__":

    init_seed = 1115
    np.random.seed(init_seed)
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    if not os.path.isdir('model/model_' + model_name + '_' + dataset):
        os.makedirs('model/model_' + model_name + '_' + dataset)
    else:
        if args.load_model < 0:
            remove_all_file('model/model_' + model_name + '_' + dataset)
    if os.path.isdir('runs/runs_' + model_name + '_' + dataset):
        if args.load_model < 0:
            remove_all_file('runs/runs_' + model_name + '_' + dataset)

    main()
