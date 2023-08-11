import basic_net as basic_net
import torch.cuda

from basic_net import fusion_img_net, transformer_resnet_wam_se_loc4, fusion_img_net_ablation_trans, fusion_img_net_cnn_weight, \
    vit_base_patch16_1024_in21k, vit_base_patch32_1024
from basic_net.DAFT_networks.vol_networks import DAFT
from basic_net.wam_block import *


class KeNet(nn.Module):
    def __init__(self, classes_num=3, basic_model='resnet', windows_attention=False, pretrain=True,
                 windows_num=3, initial_method="Uniform", k=0.8, layer_num=1):
        super(KeNet, self).__init__()
        self.basic_model = basic_model
        self.windows_attention = windows_attention
        self.wamBlock = WamBlock(in_channel=3, out_channel=3, windows_num=windows_num, initial_method=initial_method
                                 , k=k)
        if self.basic_model == 'resnet':
            self.resNet1 = basic_net.resnet34(pretrained=pretrain)
            self.resNet = list(self.resNet1.children())[:-2]
            self.features = nn.Sequential(*self.resNet)
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, classes_num)
        elif self.basic_model == 'resnet18':
            self.resNet1 = basic_net.resnet18()
            self.resNet = list(self.resNet1.children())[:-2]
            self.features = nn.Sequential(*self.resNet)
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, classes_num)
        elif self.basic_model == 'inception':
            self.inception1 = basic_net.inception_v3(pretrained=pretrain)
            self.inception = list(self.inception1.children())
            self.fc = nn.Linear(2048, classes_num)
        elif self.basic_model == 'densenet':
            self.densenet1 = basic_net.densenet121(pretrained=pretrain)
            self.densenet = nn.ModuleList(self.densenet1.children())
            self.fc = nn.Linear(1024, classes_num)
        elif self.basic_model == 'resnet-wam':
            self.resNet_WAM = basic_net.resnet34_wam(pretrained=pretrain,
                                                     windows_num=windows_num,
                                                     initial_method=initial_method, k=k, layer_num=layer_num)
            self.resNet_w = list(self.resNet_WAM.children())[:-2]
            self.features_w = nn.Sequential(*self.resNet_w)
            self.avg_pool_w = nn.AdaptiveAvgPool2d((1, 1))
            self.fc_w = nn.Linear(512, classes_num)
        elif self.basic_model == 'transformer-p16':
            self.transformer_model = vit_base_patch16_1024_in21k(num_classes=classes_num, has_logits=False)
        elif self.basic_model == 'transformer-p32':
            self.transformer_model = vit_base_patch32_1024(num_classes=classes_num)

    def forward(self, x):
        if self.windows_attention:
            x = self.wamBlock(x)
        if self.basic_model == 'resnet':
            x = self.features(x)  # 512
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
        elif self.basic_model == 'resnet18':
            x = self.features(x)  # 512
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
        elif self.basic_model == 'resnet-wam':
            x = self.features_w(x)  # 512
            x = self.avg_pool_w(x)
            x = x.view(x.size(0), -1)
            x = self.fc_w(x)
            return x
        elif self.basic_model == 'inception':
            x = self.inception[0](x)
            x = self.inception[1](x)
            x = self.inception[2](x)
            x = F.max_pool2d(x, kernel_size=3, stride=2)
            x = self.inception[3](x)
            x = self.inception[4](x)
            x = F.max_pool2d(x, kernel_size=3, stride=2)
            x = self.inception[5](x)
            x = self.inception[6](x)
            x = self.inception[7](x)
            x = self.inception[8](x)
            x = self.inception[9](x)
            x = self.inception[10](x)
            x = self.inception[11](x)
            x = self.inception[12](x)
            x = self.inception[14](x)
            x = self.inception[15](x)
            x = self.inception[16](x)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = F.dropout(x, training=self.training)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
        elif self.basic_model == 'densenet':
            features = self.densenet[0](x)  # 512
            out = F.relu(features, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
            out = self.fc(out)
            return out
        elif 'transformer' in self.basic_model:
            if '1024' not in self.basic_model:
                x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=True)
            x = self.transformer_model(x)
            return x
        elif 'trans_resnet' in self.basic_model and 'wam' not in self.basic_model:
            x = self.trans_resnet_model(x)
            return x
        elif 'trans_resnet_wam' in self.basic_model:
            x = self.trans_resnet_wam_model(x)
            return x


class KeNetMultFactorNew(nn.Module):
    def __init__(self, classes_num=3, basic_model='resnet', windows_attention=False, pretrain=True,
                 windows_num=3, initial_method="Uniform", k=0.8, layer_num=1, lb_weight=1, resize=False):
        super(KeNetMultFactorNew, self).__init__()
        self.basic_model = basic_model
        self.windows_attention = windows_attention
        self.wamBlock = WamBlock(in_channel=3, out_channel=3, windows_num=windows_num, initial_method=initial_method
                                 , k=k)
        self.factor_channel = 32
        self.resize = False
        self.lb_weight = lb_weight

        # m3: TransMUF  m2: w/o seg  m1: w/o fac
        if self.basic_model == 'm2' or self.basic_model == 'm2_n' or self.basic_model == \
                'seg_trans_resnet_wam_se_loc4_m2':
            self.MLP_factor_v1_m2 = RiskFactorNetV2(in_channel_1=2, in_channel_2=0,
                                                    out_channel=self.factor_channel)
        elif self.basic_model == 'm3' or self.basic_model == 'm3_n' or self.basic_model == 'm3_cnn' or \
                self.basic_model == 'm3_trans' or self.basic_model == 'm3_cnn_weight' or \
                self.basic_model == 'seg_trans_resnet_wam_se_loc4_m3' or self.basic_model == 'TransMUF':
            self.MLP_factor_v1_m3 = RiskFactorNetV2(in_channel_1=2, in_channel_2=3,
                                                    out_channel=self.factor_channel)
        else:
            self.MLP_factor = RiskFactorNetV2(in_channel_1=2, in_channel_2=3, out_channel=self.factor_channel)

        if self.basic_model == 'resnet':
            self.resNet1 = basic_net.resnet34(pretrained=pretrain)
            self.resNet = list(self.resNet1.children())[:-2]
            self.features = nn.Sequential(*self.resNet)
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, classes_num)
        elif self.basic_model == 'DAFT':
            self.DAFT_model = DAFT(in_channels=5, n_outputs=2, ndim_non_img=5)
        # TransMUF w/o fac
        elif self.basic_model == 'm1' or self.basic_model == 'm1_n':
            self.features_x = fusion_img_net(windows_num=windows_num,
                                             initial_method=initial_method,
                                             layer_num=layer_num,
                                             num_classes=512,
                                             input_channel=5)

            self.fc_w = FcNoFactor(512, classes_num)
        # TransMUF w/o seg
        elif self.basic_model == 'm2' or self.basic_model == 'm2_n':
            self.features_x = fusion_img_net(windows_num=windows_num,
                                             initial_method=initial_method,
                                             layer_num=layer_num,
                                             num_classes=512,
                                             input_channel=5)
            self.fc_w = FcFactor5(512, self.factor_channel, classes_num)
        # TransMUF
        elif self.basic_model == 'm3' or self.basic_model == 'm3_n' or self.basic_model == 'TransMUF':
            self.features_x = fusion_img_net(windows_num=windows_num,
                                             initial_method=initial_method,
                                             layer_num=layer_num,
                                             num_classes=512,
                                             input_channel=5)
            self.fc_w = FcFactor5(512, self.factor_channel, classes_num)
        # TransMUF w/o trans
        elif self.basic_model == 'm3_cnn_weight' or self.basic_model == 'm3_cnn':
            self.features_x = fusion_img_net_cnn_weight(windows_num=windows_num,
                                                        initial_method=initial_method,
                                                        layer_num=layer_num,
                                                        num_classes=512,
                                                        input_channel=5)
            self.fc_w = FcFactor5(512, self.factor_channel, classes_num)
        # TransMUF w/o cnn
        elif self.basic_model == 'm3_trans':
            self.features_x = fusion_img_net_ablation_trans(windows_num=windows_num,
                                                            initial_method=initial_method,
                                                            layer_num=layer_num,
                                                            num_classes=512,
                                                            input_channel=5)
            self.fc_w = FcFactor5(512, self.factor_channel, classes_num)
        # TransMUF w/o img
        elif self.basic_model == 'factor':
            self.fc = nn.Linear(self.factor_channel, classes_num)
        # Other Attempts
        elif self.basic_model == 'seg_trans_resnet_wam_se_loc4_m1':
            self.features_x = transformer_resnet_wam_se_loc4(windows_num=windows_num,
                                                             initial_method=initial_method,
                                                             layer_num=layer_num,
                                                             num_classes=512,
                                                             input_channel=5)
        elif self.basic_model == 'seg_trans_resnet_wam_se_loc4_m2':
            self.features_x = transformer_resnet_wam_se_loc4(windows_num=windows_num,
                                                             initial_method=initial_method,
                                                             layer_num=layer_num,
                                                             num_classes=512,
                                                             input_channel=5)
            self.fc_w = FcFactor5(512, self.factor_channel, classes_num)
        elif self.basic_model == 'seg_trans_resnet_wam_se_loc4_m3':
            self.features_x = transformer_resnet_wam_se_loc4(windows_num=windows_num,
                                                             initial_method=initial_method,
                                                             layer_num=layer_num,
                                                             num_classes=512,
                                                             input_channel=5)
            self.fc_w = FcFactor5(512, self.factor_channel, classes_num)

    def forward(self, x, x_seg1, x_seg2, x_non_invasive, x_invasive):
        if self.lb_weight != 1:
            x_seg2 = x_seg2 * self.lb_weight
            x_seg1 = 1 - self.lb_weight * (1 - x_seg1)

        if self.basic_model == 'DAFT':
            x = torch.cat((x, 1 - x_seg1, x_seg2), dim=1)
            fac = torch.cat((x_non_invasive, x_invasive), dim=1)
            x = self.DAFT_model(x, fac)
            return x

        if self.basic_model == 'm3' or self.basic_model == 'm3_n' or self.basic_model == 'm3_cnn' or self.basic_model \
                == 'm3_trans' or self.basic_model == 'm3_cnn_weight' or self.basic_model == 'seg_trans_resnet_wam_se_loc4_m3':
            x_factor = self.MLP_factor_v1_m3(x_non_invasive, x_invasive)
        elif self.basic_model == 'm2' or self.basic_model == 'm2_n' or self.basic_model == 'seg_trans_resnet_wam_se_loc4_m2':
            x_factor = self.MLP_factor_v1_m2(x_non_invasive, x_invasive)
        else:
            x_factor = self.MLP_factor(x_non_invasive, x_invasive)

        if self.windows_attention:
            x = self.wamBlock(x)

        if self.basic_model == 'resnet':
            x = self.features(x)  # 512
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
        elif self.basic_model == 'm1' or self.basic_model == 'm1_n':
            x = torch.cat((x, 1 - x_seg1, x_seg2), dim=1)
            x = self.features_x(x)  # 2
            x = self.fc_w(x)
            return x
        elif self.basic_model == 'm2' or self.basic_model == 'm2_n':
            x = torch.cat((x, 1 - x_seg1, x_seg2), dim=1)
            x = self.features_x(x)  # 512
            x = self.fc_w(x, x_factor)
            return x
        elif self.basic_model == 'm3' or self.basic_model == 'm3_cnn_weight' or self.basic_model == 'm3_n' \
                or self.basic_model == 'm3_cnn' or self.basic_model \
                == 'm3_trans':
            x = torch.cat((x, 1 - x_seg1, x_seg2), dim=1)
            x = self.features_x(x)  # 512
            x = self.fc_w(x, x_factor)
            return x
        elif self.basic_model == 'factor':
            x = self.fc(x_factor)
            return x
        elif self.basic_model == 'seg_trans_resnet_wam_se_loc4_m1':
            x = torch.cat((x, 1 - x_seg1, x_seg2), dim=1)
            x = self.features_x(x)  # 2
            return x
        elif self.basic_model == 'seg_trans_resnet_wam_se_loc4_m2':
            x = torch.cat((x, 1 - x_seg1, x_seg2), dim=1)
            x = self.features_x(x)  # 512
            x = self.fc_w(x, x_factor)
            return x
        elif self.basic_model == 'seg_trans_resnet_wam_se_loc4_m3':
            x = torch.cat((x, 1 - x_seg1, x_seg2), dim=1)
            x = self.features_x(x)  # 512
            x = self.fc_w(x, x_factor)
            return x


class RiskFactorNetV2(nn.Module):
    def __init__(self, in_channel_1, in_channel_2, out_channel=128):
        super(RiskFactorNetV2, self).__init__()
        self.in_channel_2 = in_channel_2
        if in_channel_1 != 0:
            self.MLP1 = nn.Sequential(
                nn.Linear(in_channel_1, 8),
                nn.BatchNorm1d(8),
                nn.ReLU(inplace=True),
            )

        if in_channel_2 != 0:
            self.MLP2 = nn.Sequential(
                nn.Linear(in_channel_2, 8),
                nn.BatchNorm1d(8),
                nn.ReLU(inplace=True),
            )

        self.MLP_Mix = nn.Sequential(
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Linear(32, out_channel),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(inplace=True),
        )

        self.MLP_Mix_1 = nn.Sequential(
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),

            nn.Linear(16, out_channel),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(0.5)

    def forward(self, x_fac1, x_fac2):
        x_fac1 = self.MLP1(x_fac1)
        if self.in_channel_2 != 0:
            x_fac2 = self.MLP2(x_fac2)
            x = torch.cat((x_fac1, x_fac2), dim=1)
            x = self.MLP_Mix(x)
        else:
            x = x_fac1
            x = self.MLP_Mix_1(x)
        x = self.dropout(x)
        return x


class FcFactor(nn.Module):
    def __init__(self, in_channel, factor_channel, out_channel):
        super(FcFactor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channel + factor_channel, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Linear(64, out_channel),
        )

    def forward(self, x, factor):
        x = torch.cat((x, factor), dim=1)
        x = self.fc(x)
        return x


class FcFactor5(nn.Module):
    def __init__(self, in_channel, factor_channel, out_channel):
        super(FcFactor5, self).__init__()
        self.ln1 = nn.Sequential(
            nn.Linear(in_channel, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 + factor_channel, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Linear(64, out_channel),
        )

    def forward(self, x, factor):
        x = self.ln1(x)
        x = torch.cat((x, factor), dim=1)
        x = self.fc(x)
        return x


class FcNoFactor(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FcNoFactor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channel, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Linear(64, out_channel),
        )

    def forward(self, x):
        x = self.fc(x)
        return x
