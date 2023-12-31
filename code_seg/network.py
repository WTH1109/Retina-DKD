import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import numpy as np
import math
import os
import glob
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import math
import os
import torchvision.models as models

import torch.cuda



""" classification"""



class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Multiscale_layer(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Multiscale_layer, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, int(ch_out // 3), kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(int(ch_out // 3)),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_in, int(ch_out // 3), kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(int(ch_out // 3)),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(ch_in, int(ch_out // 3), kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm2d(int(ch_out // 3)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = torch.cat((x1, x2), 1)
        x = torch.cat((x3, x), 1)
        return x


class Seg_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=2):
        super(Seg_Net, self).__init__()
        self.Conv0 = Multiscale_layer(ch_in=img_ch, ch_out=36)

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(ch_in=36, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
        self.poly1 = nn.Conv2d(1024, 2, kernel_size=1)
        self.poly2 = nn.Conv2d(512, 2, kernel_size=1)
        self.poly3 = nn.Conv2d(256, 2, kernel_size=1)
        self.poly4 = nn.Conv2d(128, 2, kernel_size=1)
        self.poly5 = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        # encoding path
        x = self.Conv0(x)

        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        self.out1_ori = self.poly1(x5)
        self.out2_ori = self.poly2(d5)
        self.out3_ori = self.poly3(d4)
        self.out4_ori = self.poly4(d3)
        self.out5_ori = self.poly5(d2)

        self.out1 = F.interpolate(self.out1_ori,
                                  size=(d2.shape[2], d2.shape[3]))
        self.out2 = F.interpolate(self.out2_ori,
                                  size=(d2.shape[2], d2.shape[3]))
        self.out3 = F.interpolate(self.out3_ori,
                                  size=(d2.shape[2], d2.shape[3]))
        self.out4 = F.interpolate(self.out4_ori,
                                  size=(d2.shape[2], d2.shape[3]))
        self.out5 = F.interpolate(self.out5_ori,
                                  size=(d2.shape[2], d2.shape[3]))

        self.out = 0.125 * self.out1 + 0.25 * self.out2 + 0.25 * self.out3 + 0.5*self.out4+self.out5
        # self.out =  0.25 * self.out2 + 0.25 * self.out3 + 0.5 * self.out4 + self.out5
        # self.out =  0.25 * self.out3 + 0.5 * self.out4 + self.out5
        # self.out =0.5 * self.out4 + self.out5
        # self.out = self.out5
        return self.out
        # d1 = self.Conv_1x1(d2)
        # return d1






