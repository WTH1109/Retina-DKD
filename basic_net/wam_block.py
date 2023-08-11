from torch import nn
from torch.nn.parameter import Parameter
import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F


class WamBlock(nn.Module):
    """
    Args:
        in_channel: input channel num
        out_channel: output channel num
        windows_num: The number of convolution block
        initial_method: How to initial the windows weight(uniform","linear","exp","random")
        k: If you use the method of "Linear" or "Exp" initial method, this parameter can change the method strength
            For example:
            In "Linear" method, k means the gradient, 3*3 window gets the highest weight
            In "Exp" method, we use the function exp(-k*i)/sum(exp(-k*i)) to initial the windows-weight
                you can samply understand it as follows: The bigger the k, the higher weight of 3*3 windows
    """

    def __init__(self, in_channel, out_channel, windows_num=3, initial_method="uniform", block="default", k=0.8
                 , stride=1, down_sample=None):
        super(WamBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.windows_num = windows_num

        if initial_method == "linear":
            t_tensor = torch.arange(1, windows_num + 1)
            self.initial_tensor = (-t_tensor * k / windows_num + 1)
            self.initial_tensor = self.initial_tensor / torch.sum(self.initial_tensor)
        elif initial_method == "exp":
            t_tensor = torch.arange(1, windows_num + 1)
            self.initial_tensor = torch.exp(-t_tensor * k)
            self.initial_tensor = self.initial_tensor / torch.sum(self.initial_tensor)
        elif initial_method == "random":
            self.initial_tensor = torch.rand(windows_num)
            self.initial_tensor = self.initial_tensor / torch.sum(self.initial_tensor)
        else:
            self.initial_tensor = torch.ones(windows_num)
            self.initial_tensor = torch.div(self.initial_tensor, windows_num)
        self.windows_weight = Parameter(self.initial_tensor, requires_grad=True)
        self.wam_block = {}
        for i in range(0, self.windows_num):
            if block == "default":
                self.wam_block['cb_' + str(2 * i + 3)] = conv_block(ch_in=self.in_channel, ch_out=self.out_channel,
                                                                    kernel_size=2 * i + 3)
            elif block == "basic_block":
                self.wam_block['cb_' + str(2 * i + 3)] = BasicBlock(inplanes=self.in_channel, planes=self.out_channel
                                                                    , stride=stride, down_sample=down_sample)
        self.wam_block = nn.ModuleDict(self.wam_block)

    def forward(self, x):
        cache = x
        cache_w = {'cw_' + str(2 * 0 + 3): cache * self.windows_weight[0]}
        x = self.wam_block['cb_' + str(2 * 0 + 3)](cache_w['cw_' + str(2 * 0 + 3)])
        for i in range(1, self.windows_num):
            cache_w['cw_' + str(2 * i + 3)] = cache * self.windows_weight[i]
            x = x + self.wam_block['cb_' + str(2 * i + 3)](cache_w['cw_' + str(2 * i + 3)])
        return x


class WamBlock_alpha_mask(nn.Module):
    """
    Args:
        in_channel: input channel num
        out_channel: output channel num
        windows_num: The number of convolution block
        initial_method: How to initial the windows weight(uniform","linear","exp","random")
        k: If you use the method of "Linear" or "Exp" initial method, this parameter can change the method strength
            For example:
            In "Linear" method, k means the gradient, 3*3 window gets the highest weight
            In "Exp" method, we use the function exp(-k*i)/sum(exp(-k*i)) to initial the windows-weight
                you can samply understand it as follows: The bigger the k, the higher weight of 3*3 windows
    """

    def __init__(self, in_channel, out_channel, windows_num=3, initial_method="uniform", block="default", k=0.8
                 , stride=1, down_sample=None, alpha=0.5, if_mask=False):
        super(WamBlock_alpha_mask, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.windows_num = windows_num
        self.alpha = alpha

        if initial_method == "linear":
            t_tensor = torch.arange(1, windows_num + 1)
            self.initial_tensor = (-t_tensor * k / windows_num + 1)
            self.initial_tensor = self.initial_tensor / torch.sum(self.initial_tensor)
        elif initial_method == "exp":
            t_tensor = torch.arange(1, windows_num + 1)
            self.initial_tensor = torch.exp(-t_tensor * k)
            self.initial_tensor = self.initial_tensor / torch.sum(self.initial_tensor)
        elif initial_method == "random":
            self.initial_tensor = torch.rand(windows_num)
            self.initial_tensor = self.initial_tensor / torch.sum(self.initial_tensor)
        else:
            self.initial_tensor = torch.ones(windows_num)
            self.initial_tensor = torch.div(self.initial_tensor, windows_num)
        self.windows_weight = Parameter(self.initial_tensor, requires_grad=True)
        self.wam_block = {}
        for i in range(0, self.windows_num):
            if block == "default":
                self.wam_block['cb_' + str(2 * i + 3)] = conv_block_alpha_mask(ch_in=self.in_channel,
                                                                               ch_out=self.out_channel,
                                                                               kernel_size=2 * i + 3, alpha=alpha,
                                                                               if_mask=if_mask)
            elif block == "basic_block":
                self.wam_block['cb_' + str(2 * i + 3)] = BasicBlock_alpha_mask(inplanes=self.in_channel,
                                                                               planes=self.out_channel
                                                                               , stride=stride, down_sample=down_sample,
                                                                               alpha=alpha, if_mask=if_mask)
        self.wam_block = nn.ModuleDict(self.wam_block)

    def forward(self, x, mask1, mask2):
        cache = x
        cache_w = {'cw_' + str(2 * 0 + 3): cache * self.windows_weight[0]}
        x, _, _ = self.wam_block['cb_' + str(2 * 0 + 3)](cache_w['cw_' + str(2 * 0 + 3)], mask1, mask2)
        for i in range(1, self.windows_num):
            cache_w['cw_' + str(2 * i + 3)] = cache * self.windows_weight[i]
            x_t, _, _ = self.wam_block['cb_' + str(2 * i + 3)](cache_w['cw_' + str(2 * i + 3)], mask1, mask2)
            x = x + x_t
        return x, mask1, mask2


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=1, padding=int((kernel_size - 1) / 2), bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=kernel_size, stride=1, padding=int((kernel_size - 1) / 2), bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class conv_block_alpha_mask(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, alpha, if_mask=False):
        super(conv_block_alpha_mask, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=1, padding=int((kernel_size - 1) / 2), bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=kernel_size, stride=1, padding=int((kernel_size - 1) / 2), bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.if_mask = if_mask
        self.alpha = alpha

    def mask_x(self, x, mask1, mask2):
        b, c, h, w = x.shape
        mask1 = F.interpolate(mask1, size=(h, w), mode='bilinear', align_corners=True)
        mask2 = F.interpolate(mask2, size=(h, w), mode='bilinear', align_corners=True)
        x = x * (self.alpha + (1 - self.alpha) / 2 * mask1 + (1 - self.alpha) / 2 * mask2)
        return x

    def forward(self, x, mask1, mask2):
        x = self.conv1(x)
        if self.if_mask:
            x = self.mask_x(x, mask1, mask2)
        x = self.conv2(x)
        if self.if_mask:
            x = self.mask_x(x, mask1, mask2)
        return x, mask1, mask2


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, down_sample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlock_alpha_mask(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, down_sample=None, alpha=0.5, if_mask=False):
        super(BasicBlock_alpha_mask, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.down_sample = down_sample
        self.stride = stride
        self.alpha = alpha
        self.if_mask = if_mask

    def mask_x(self, x, mask1, mask2):
        b, c, h, w = x.shape
        mask1 = F.interpolate(mask1, size=(h, w), mode='bilinear', align_corners=True)
        mask2 = F.interpolate(mask2, size=(h, w), mode='bilinear', align_corners=True)
        x = x * (self.alpha + (1 - self.alpha) / 2 * mask1 + (1 - self.alpha) / 2 * mask2)
        return x

    def forward(self, x, mask1, mask2):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.if_mask:
            out = self.mask_x(out, mask1, mask2)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.if_mask:
            out = self.mask_x(out, mask1, mask2)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out, mask1, mask2


if __name__ == "__main__":
    writer = SummaryWriter(logdir="model_graph/wam")
    input_x = torch.zeros((64, 3, 1024, 1024))
    w_net = WamBlock(in_channel=3, out_channel=5, windows_num=4, initial_method="linear")
    writer.add_graph(w_net, input_x)
    writer.close()
    print("done")
