import os

import yaml


def run_py(python_name, dataset, model_name, xlsx_name, dataset_num, model_name_save, basic_model, gpu, wam,
           windows_num, el, er, layer_num, mask_r=None, fold_name=None, lb_weight=1):
    print(dataset)
    basic_order = 'python ' + python_name + ' -s ' + dataset + ' -m ' + model_name + ' -x ' + xlsx_name + ' -d ' + str(
        dataset_num) + ' -mnp ' + model_name_save + ' -b ' + basic_model + ' -g ' + str(
        gpu) + ' -w ' + wam + ' -n ' + str(windows_num) + ' -el ' + str(el) + ' -er ' + str(er) + ' -l ' + str(
        layer_num)
    if fold_name is not None:
        order = basic_order + ' -df ' + fold_name
    if lb_weight != 1:
        order = order + ' -lb ' + str(lb_weight)
    print(order)
    os.system(order)


with open('config/test_config.yaml') as file:
    config = yaml.load(file, Loader=yaml.SafeLoader)

python_name_set = [
    'test_isolate_all.py',  # 0
    'test_fusion.py',       # 1
    'test_fusion_pat.py',   # 2
    'grad-cam-fusion.py',   # 3
]

lb_weight = config['lb_weight']
select_num = config['select_num']
python_name_r = python_name_set[select_num]
basic_model_r = config['basic_model']
mask_set = ['images', 'disk', 'lesion', 'HbA1C', 'Hb', 'SBP', 'DM', 'hematuresis']
fold_name = config['dataset_supple']

if select_num == 10:
    mask = mask_set[0]
else:
    mask = None

gpu_r = config['gpu']
wam_r = 'False'
windows_num_r = config['win_num']
layer_num_r = 1
pretrain = False
ep_true = config['ep_true']
ep = ep_true + 7
model_name_r = config['basic_model'] + str(ep_true)  # sheet name
ep_n = 1
el_r, er_r = ep - 3 - ep_n * 4, ep - 3

model_name_save_r = config['model_name_save_r']

xlsx_name_r = config['xlsx_name_r']
dataset_r = ['main', 'Prospective', 'Multi_center', 'Non_standard']
dataset = config['dataset_select']

if select_num == 3:
    xlsx_name_r = 'runs_result_gradcam.xls'
    dataset = [0]

for i in dataset:
    run_py(python_name_r, dataset_r[i], model_name_r, xlsx_name_r, dataset_num=i + 1, model_name_save=model_name_save_r,
           basic_model=basic_model_r,
           gpu=gpu_r, wam=wam_r, windows_num=windows_num_r, el=el_r, er=er_r, layer_num=layer_num_r, mask_r=mask,
           fold_name=fold_name, lb_weight=lb_weight)
