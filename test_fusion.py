import matplotlib
from data_pre_process.data_process import DrdatasetMultidataFactor5
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import torch.cuda
from tqdm import tqdm
from sklearn import metrics
import argparse
import pandas as pd

from network import KeNetMultFactorNew
from test_file.bootstrap_ci import my_bootstrap_ci_model
from data_pre_process.xlsx_process import test_data_write_xlsx

parser = argparse.ArgumentParser(description='test_fusion')
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

if 1:
    device_ids = [pars.gpu]
    basic_model = pars.basic_model  # inception fold1 797 densenet fold0 793 resnet fold0 793
    fold = '0'
    test_epoch = range(pars.ep_left, pars.ep_right, 4)

if pars.wam == 'True' or pars.wam == 'true':
    windows_attention = True
else:
    windows_attention = False

net = KeNetMultFactorNew(classes_num=2, basic_model=basic_model, windows_attention=windows_attention,
                         pretrain=False
                         , windows_num=pars.win_num, initial_method="Uniform", k=0.8,
                         layer_num=pars.layer, lb_weight=pars.lamb).cuda(
    device_ids[0])

img_size = 1024
num_class = 2
dataset = 'DKD'
num_thread = 8

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

net = torch.nn.DataParallel(net, device_ids)

Total_Acc_ci = []
Total_Spec_ci = []
Total_Sen_ci = []
Total_F1_Score_ci = []
Total_AUC_ci = []

Total_Acc = []
Total_Spec = []
Total_Sen = []
Total_F1_Score = []
Total_AUC = []

distribution_scatter_0 = []
distribution_scatter_1 = []


def main():
    dis_show = 0
    print("Waiting Test!")
    label_all_allfold = []
    predicted_all_allfold = []
    with open('isolate_test/cls_metrics/metrics_' + model_name + '_' + dataset + '_isolate.txt', "w+") as f:
        for ep in test_epoch:
            ep_s = str(ep)
            while len(ep_s) < 3:
                ep_s = '0' + ep_s
            a = torch.load(model_dir + 'net_' + ep_s + '.pth', map_location='cpu')
            net.load_state_dict(a)
            if pars.dataset == '_all':
                phase = 'Train'
            else:
                phase = 'Test'
            dr_dataset_test = DrdatasetMultidataFactor5(
                root_img=set_dir,
                root_seg1=disk_dir,
                root_seg2=lesion_dir,
                xlsx_path=xlsx_path,
                phase=phase,
                img_size=img_size, num_class=num_class, transform=False, isolate=isolate, if_after=if_after)
            loader_test = DataLoader(dr_dataset_test, batch_size=1, num_workers=num_thread, shuffle=False)
            test_bar = tqdm(loader_test)

            with torch.no_grad():
                label_all = []
                predicted_all = []
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
                    outputs = torch.softmax(outputs, dim=1)
                    predicted_all.append(outputs.detach().cpu().numpy()[0][1])
                    _, predicted = torch.max(outputs.data, 1)
                    # predicted = torch.gt(outputs.data, 0.7)

                    labels = labels.cpu().numpy()
                    label_all.append(labels[0])
                    predicted = predicted.cpu().numpy()
                    if labels[0] == 0:
                        distribution_scatter_0.append(torch.softmax(outputs.cpu(), dim=1).numpy()[0][1])
                    else:
                        distribution_scatter_1.append(torch.softmax(outputs.cpu(), dim=1).numpy()[0][1])
                    for i_test in range(1):
                        if labels[i_test] == 1 and predicted[i_test] == 1:
                            tp += 1
                        if labels[i_test] == 1 and predicted[i_test] == 0:
                            fn += 1
                        if labels[i_test] == 0 and predicted[i_test] == 1:
                            fp += 1
                        if labels[i_test] == 0 and predicted[i_test] == 0:
                            tn += 1

                Acc = (tp + tn) / (tp + tn + fp + fn)
                print('epoc:%d acc:%f' % (ep, Acc * 100))
                Sen = (tp) / (tp + fn)  # recall
                Spec = (tn) / (tn + fp)

                if tp + fp == 0:
                    precision = 0
                else:
                    precision = (tp) / (tp + fp)
                if tn + fn == 0:
                    npv = 0
                else:
                    npv = (tn) / (tn + fn)
                recall = Sen
                if precision + recall == 0:
                    f1_score = 0
                else:
                    f1_score = (2 * precision * recall) / (precision + recall)
                label_all_allfold = label_all_allfold + label_all
                predicted_all_allfold = predicted_all_allfold + predicted_all
                AUC = metrics.roc_auc_score(y_true=np.array(label_all), y_score=np.array(predicted_all))

                ACC_low, ACC_high, SEN_low, SEN_high, SPEC_low, SPEC_high, F1_SCORE_low, F1_SCORE_high, \
                    AUC_low, AUC_high = my_bootstrap_ci_model(
                    predicted_all, label_all)

                Total_Acc_ci.append((Acc, ACC_low, ACC_high))
                Total_Sen_ci.append((Sen, SEN_low, SEN_high))
                Total_Spec_ci.append((Spec, SPEC_low, SPEC_high))
                Total_AUC_ci.append((AUC, AUC_low, AUC_high))
                Total_F1_Score_ci.append((f1_score, F1_SCORE_low, F1_SCORE_high))

                Total_Acc.append(Acc)
                Total_Sen.append(Sen)
                Total_Spec.append(Spec)
                Total_AUC.append(AUC)
                Total_F1_Score.append(f1_score)

                f.write('ep' + str(ep))
                f.write(' ')
                f.flush()
                f.write('Acc=: %.3f%%' % Acc)
                f.write('  ')
                f.flush()
                f.write('Sen=: %.3f%%' % Sen)
                f.write('  ')
                f.flush()
                f.write('Spec=: %.3f%%' % Spec)
                f.write('  ')
                f.flush()
                f.write('precision=: %.3f%%' % precision)
                f.write('  ')
                f.flush()
                f.write('npv=: %.3f%%' % npv)
                f.write('  ')
                f.flush()
                f.write('f1_score=: %.3f%%' % f1_score)
                f.write('  ')
                f.flush()
                f.write('AUC=: %.3f%%' % AUC)
                f.write('\n')
                f.flush()

                if dis_show == 0:
                    bins = 30
                    matplotlib.use('Agg')
                    plt.figure()
                    plt.hist(np.array(distribution_scatter_0), bins=bins, color='blue')
                    plt.hist(np.array(distribution_scatter_1), bins=bins, color='red')
                    plt.savefig(
                        './isolate_test/cls_metrics/metrics_' + model_name + '_distribution/' + dataset + '/mix.jpg')
                    plt.figure()
                    plt.hist(np.array(distribution_scatter_1), bins=bins, color='red')
                    plt.savefig(
                        './isolate_test/cls_metrics/metrics_' + model_name + '_distribution/' + dataset + '/positive.jpg')
                    plt.figure()
                    plt.hist(np.array(distribution_scatter_0), bins=bins, color='blue')
                    plt.savefig(
                        './isolate_test/cls_metrics/metrics_' + model_name + '_distribution/' + dataset + '/negative.jpg')
                else:
                    dis_show = 1

                # roc curve

                fpr, tpr, _ = metrics.roc_curve(y_true=np.array(label_all), y_score=np.array(predicted_all))
                data_roc = {'label': np.array(label_all),
                            'pre': np.array(predicted_all)}
                roc_curve = pd.DataFrame(data=data_roc)
                roc_data_save_path = 'excel_data/roc_data/'
                if not os.path.isdir('excel_data'):
                    os.mkdir('excel_data')
                if not os.path.isdir(roc_data_save_path):
                    os.mkdir(roc_data_save_path)
                roc_data_save_dir = roc_data_save_path + model_name + '/'
                if not os.path.isdir(roc_data_save_dir):
                    os.mkdir(roc_data_save_dir)
                roc_curve.to_excel(roc_data_save_dir + dataset + '.xlsx')
                # plt.rcParams["font.family"] = "Arial"
                # plt.rcParams["font.weight"] = "bold"
                # plt.rcParams["axes.labelweight"] = "bold"
                # plt.figure()
                # plt.plot(fpr, tpr,
                #          label='ROC curve of fold ' + str(fold[0]) + ' (AUC=%.3f' % AUC + ')',
                #          color='black', linestyle='-', linewidth=4)
                #
                # plt.plot([0, 1], [0, 1], 'k--', lw=2)
                # plt.xlim([0.0, 1.0])
                # plt.ylim([0.0, 1.05])
                # plt.fill_between(fpr, tpr, facecolor='gray', alpha=0.5)
                # font_axis_name = {'fontsize': 22, 'weight': 'bold'}
                # plt.xlabel('1 - Specificity', font_axis_name)
                # plt.ylabel('Sensitivity', font_axis_name)
                # plt.title('ROC curves', font_axis_name)
                # plt.legend(loc="lower right")
                # plt.grid()
                # plt.savefig('isolate_test/ROC/isolate_' + model_name + '_roc_fold' + fold[0] + str(ep) + '.tif')

    ave_acc = np.average(Total_Acc)
    var_acc = np.var(Total_Acc)

    ave_sen = np.average(Total_Sen)
    var_sen = np.var(Total_Sen)

    ave_spec = np.average(Total_Spec)
    var_spec = np.var(Total_Spec)

    ave_f1_score = np.average(Total_F1_Score)
    var_f1_score = np.var(Total_F1_Score)

    ave_auc = np.average(Total_AUC)
    var_auc = np.var(Total_AUC)

    print(model_name + ':')
    print(dataset + '--------')
    print('ave_acc: %.2f%%    var_acc: %.2f%%   ci=[%.2f%% , %.2f%%]' % (
        ave_acc * 100, var_acc * 100, Total_Acc_ci[0][1] * 100, Total_Acc_ci[0][2] * 100))
    print('ave_sen: %.2f%%    var_sen: %.2f%%   ci=[%.2f%% , %.2f%%]' % (
        ave_sen * 100, var_sen * 100, Total_Sen_ci[0][1] * 100, Total_Sen_ci[0][2] * 100))
    print('ave_spec: %.2f%%   var_spec: %.2f%%  ci=[%.2f%% , %.2f%%]' % (
        ave_spec * 100, var_spec * 100, Total_Spec_ci[0][1] * 100, Total_Spec_ci[0][2] * 100))
    print('ave_f1sc: %.2f%%   var_f1sc: %.2f%%  ci=[%.2f%% , %.2f%%]' % (
        ave_f1_score * 100, var_f1_score * 100, Total_F1_Score_ci[0][1] * 100, Total_F1_Score_ci[0][2] * 100))
    print('ave_auc: %.2f%%    var_auc: %.2f%%   ci=[%.2f%% , %.2f%%]' % (
        ave_auc * 100, var_auc * 100, Total_AUC_ci[0][1] * 100, Total_AUC_ci[0][2] * 100))

    data_all = [ave_acc, var_acc, ave_sen, var_sen, ave_spec, var_spec, ave_f1_score, var_f1_score, ave_auc, var_auc]
    test_data_write_xlsx(pars.xlsx_name, pars.model_name, pars.set, pars.dataset_num, data_all, pars.model_name_path,
                         len(test_epoch))


if __name__ == "__main__":

    init_seed = 1115
    np.random.seed(init_seed)
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    if not os.path.exists('./isolate_test'):
        os.mkdir('./isolate_test')
    if not os.path.exists('./isolate_test/ROC/'):
        os.mkdir('./isolate_test/ROC/')
    if not os.path.exists('./isolate_test/cls_metrics/'):
        os.mkdir('./isolate_test/cls_metrics/')
    if not os.path.exists('./isolate_test/cls_metrics/metrics_' + model_name + '_distribution/'):
        os.mkdir('./isolate_test/cls_metrics/metrics_' + model_name + '_distribution/')
    if not os.path.exists('./isolate_test/cls_metrics/metrics_' + model_name + '_distribution/' + dataset):
        os.mkdir('./isolate_test/cls_metrics/metrics_' + model_name + '_distribution/' + dataset)
    main()
