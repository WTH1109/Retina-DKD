"""
@Author : wth
@Time : 2023/9/23 10:20
@File : Diagnosis_demo.py
@Comments : 
"""
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from skimage import io
from PIL import Image
from data_pre_process import get_id
from grad_cam_model import GradCAM, get_nn_target_layer
from grad_cam_model.cam_utils import show_cam_on_image
from network import KeNetMultFactorNew

img_name = '9-C853904-70408.jpg'
img_id = get_id(img_name)

# 三张图片存储的位置
img_path = '/mnt/ssd4/wengtaohan/Retina-DKD/data/cls_dkd/Train/DN'
disk_path = '/mnt/ssd4/wengtaohan/Retina-DKD/data/seg/disk/Train/DN'
lesion_path = '/mnt/ssd4/wengtaohan/Retina-DKD/data/seg/lesion/Train/DN'

# 临床指标存储位置
factor_path = '/mnt/ssd4/wengtaohan/Retina-DKD/data/risk_factor_5.xlsx'

# 模型存放位置
model_dir = '/mnt/ssd4/wengtaohan/Retina-DKD/model/transMUF/net_25.pth'

def main():
    # 指定运算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = True if torch.cuda.is_available() else False

    # 设置模型
    basic_model = 'TransMUF'
    net = KeNetMultFactorNew(classes_num=2, basic_model=basic_model, windows_attention=False,
                             pretrain=False
                             , windows_num=3, initial_method="Uniform", k=0.8,
                             layer_num=1, lb_weight=1).to(device)
    net = torch.nn.DataParallel(net, ['cuda:0'])

    # 加载模型参数
    model_paras = torch.load(model_dir, map_location='cpu')
    net.load_state_dict(model_paras)

    with torch.no_grad():
        # 模型运算
        net.eval()
        images, seg1, seg2, non_inv_fac, inv_fac = get_data(img_name, img_path, disk_path, lesion_path, factor_path)
        outputs = net(images.to(device), seg1.to(device), seg2.to(device), non_inv_fac.to(device), inv_fac.to(device))

        # 计算预测值
        outputs = torch.softmax(outputs, dim=1)
        pre = outputs.cpu().detach().numpy()[0, 1]
        pre_label = 1 if pre > 0.5 else 0

    # 可视化模型
    target_layer = get_nn_target_layer(net, basic_model)
    cam = GradCAM(model=net, target_layers=target_layer, use_cuda=use_cuda)
    grayscale_cam = cam(input_tensor=[images, seg1, seg2, non_inv_fac, inv_fac], target_category=pre_label)[0, :]
    img = images.cpu().numpy()[0]
    img = np.transpose(img, (1, 2, 0))
    visualization = show_cam_on_image(img.astype(dtype=np.float32), grayscale_cam, use_rgb=True)

    # 打印可视化结果
    plt.imshow(visualization)
    plt.show()

    # 打印诊断结果
    print(pre)
    if pre > 0.5:
        print('NDRD')
    else:
        print('DN')


def get_data(_img_name, _img_path, _disk_path, _lesion_path, _factor_path):
    image = io.imread(os.path.join(_img_path, _img_name))
    seg_1 = io.imread(os.path.join(_disk_path, _img_name))
    seg_2 = io.imread(os.path.join(_lesion_path, _img_name))

    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = image.astype(np.dtype(np.float32))
    seg_1 = seg_reshape(seg_1, normalize=True)
    seg_2 = seg_reshape(seg_2, normalize=True)

    id_usr = get_id(_img_name)
    invasive, non_invasive = get_sheet_data_from_id(id_usr, _factor_path)

    return torch.from_numpy(np.array(image)).unsqueeze(0), torch.from_numpy(np.array(seg_1)).unsqueeze(0),\
        torch.from_numpy(np.array(seg_2)).unsqueeze(0), \
        torch.from_numpy(np.array(non_invasive)).to(torch.float32).unsqueeze(0), \
        torch.from_numpy(np.array(invasive)).to(torch.float32).unsqueeze(0)

def pil_loader(image_path, if_mask=False):
    with open(image_path, 'rb') as f:
        img = Image.open(f)
        if if_mask:
            img = np.array(img)
            img[img > 80] = 255
            img[img <= 80] = 0
            img = Image.fromarray(img.astype('uint8'))
        return img.convert('RGB')

def seg_reshape(seg, normalize=True):
    if normalize:
        seg = seg / 255.0
    dim_num = seg.ndim
    if dim_num == 3:
        seg = np.mean(seg, 2)
    m, n = seg.shape
    seg = np.reshape(seg, (1, m, n))
    seg = seg.astype(np.dtype(np.float32))
    return seg

def get_sheet_data_from_id(id_num, xlsx_path):
    data_xlsx = pd.read_excel(xlsx_path).iloc[:, 0:8]
    data_xlsx.fillna(0, inplace=True)
    usr_dat = data_xlsx[data_xlsx.iloc[:, 1] == id_num]
    press_r = usr_dat.iloc[0, 3] / 202.0
    course_r = usr_dat.iloc[0, 4] / 564.0
    mashing_r = usr_dat.iloc[0, 5] / 13.8
    protein_r = usr_dat.iloc[0, 6] / 207.0
    eGFR_r = usr_dat.iloc[0, 7] / 132.0

    _non_invasive_r = [press_r, course_r]
    _invasive_r = [mashing_r, protein_r, eGFR_r]

    return _invasive_r, _non_invasive_r


if __name__ == '__main__':
    main()