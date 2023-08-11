import torch.utils.model_zoo as model_zoo

from seg_net import feature_match
from basic_net.wam_block import *

__all__ = ['ResNet_wam', 'resnet18', 'resnet34_wam', 'resnet50', 'resnet101',
           'resnet152', 'resnet34_wam_alpha_mask']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet_wam(nn.Module):

    def __init__(self, block, block_wam, layers, input_channel=3, num_classes=1000, zero_init_residual=False,
                 windows_num=3, initial_method="Uniform", k=0.8, layer_num=1):
        super(ResNet_wam, self).__init__()
        self.inplanes = 64
        # 3 * 1024 * 1024
        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        # 64 * 512 * 512
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 64 * 256 * 256
        self.layer1 = self._make_layer(block, block_wam, planes=64, block_num=layers[0], stride=1,
                                       block_type='wam_half', windows_num=windows_num, initial_method=initial_method,
                                       k=k)
        # 64 * 256 * 256
        if layer_num >= 2:
            self.layer2 = self._make_layer(block, block_wam, planes=128, block_num=layers[1], stride=2,
                                           block_type='wam_half', windows_num=windows_num,
                                           initial_method=initial_method,
                                           k=k)
        else:
            self.layer2 = self._make_layer(block, block_wam, planes=128, block_num=layers[1], stride=2)
        # 128 *
        if layer_num >= 3:
            self.layer3 = self._make_layer(block, block_wam, planes=256, block_num=layers[2], stride=2,
                                           block_type='wam_half', windows_num=windows_num,
                                           initial_method=initial_method,
                                           k=k)
        else:
            self.layer3 = self._make_layer(block, block_wam, planes=256, block_num=layers[2], stride=2)

        if layer_num >= 4:
            self.layer4 = self._make_layer(block, block_wam, planes=512, block_num=layers[3], stride=2,
                                           block_type='wam_half', windows_num=windows_num,
                                           initial_method=initial_method,
                                           k=k)
        else:
            self.layer4 = self._make_layer(block, block_wam, planes=512, block_num=layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, block_wam, planes, block_num,
                    stride=1, block_type='default', windows_num=3, initial_method="Uniform", k=0.8):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if block_type == 'default' or block_type == 'wam_half':
            layers.append(block(self.inplanes, planes, stride, downsample))
        elif block_type == 'wam_whole':
            layers.append(block_wam(self.inplanes, planes, windows_num=windows_num, initial_method=initial_method
                                    , block='basic_block', k=k, stride=stride, downsample=downsample))

        self.inplanes = planes * block.expansion
        for _ in range(1, block_num):
            if block_type == 'default':
                layers.append(block(self.inplanes, planes))
            elif block_type == 'wam_half' or block_type == 'wam_whole':
                layers.append(block_wam(self.inplanes, planes, windows_num=windows_num, initial_method=initial_method
                                        , block='basic_block', k=k, stride=stride, down_sample=downsample))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNet_alpha_mask(nn.Module):

    def __init__(self, block, block_wam, layers, input_channel=3, num_classes=1000, zero_init_residual=False,
                 windows_num=3, initial_method="Uniform", k=0.8, layer_num=1, alpha=0.5):
        super(ResNet_alpha_mask, self).__init__()
        self.inplanes = 64
        self.alpha = alpha
        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = make_layer_alpha_mask(64, block, block_wam, planes=64, block_num=layers[0], stride=1,
                                            block_type='wam_half', windows_num=windows_num,
                                            initial_method=initial_method,
                                            k=k, alpha=alpha, if_mask=True)
        if layer_num >= 2:
            self.layer2 = make_layer_alpha_mask(64, block, block_wam, planes=128, block_num=layers[1], stride=2,
                                                block_type='wam_half', windows_num=windows_num,
                                                initial_method=initial_method,
                                                k=k, alpha=alpha)
        else:
            self.layer2 = make_layer_alpha_mask(64, block, block_wam, planes=128, block_num=layers[1], stride=2,
                                                alpha=alpha)

        if layer_num >= 3:
            self.layer3 = make_layer_alpha_mask(128, block, block_wam, planes=256, block_num=layers[2], stride=2,
                                                block_type='wam_half', windows_num=windows_num,
                                                initial_method=initial_method,
                                                k=k, alpha=alpha)
        else:
            self.layer3 = make_layer_alpha_mask(128, block, block_wam, planes=256, block_num=layers[2], stride=2,
                                                alpha=alpha)

        if layer_num >= 4:
            self.layer4 = make_layer_alpha_mask(256, block, block_wam, planes=512, block_num=layers[3], stride=2,
                                                block_type='wam_half', windows_num=windows_num,
                                                initial_method=initial_method,
                                                k=k, alpha=alpha)
        else:
            self.layer4 = make_layer_alpha_mask(256, block, block_wam, planes=512, block_num=layers[3], stride=2,
                                                alpha=alpha)
        self.avgpool_w = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_w = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def mask_x(self, x, mask1, mask2):
        b, c, h, w = x.shape
        mask1 = F.interpolate(mask1, size=(h, w), mode='bilinear', align_corners=True)
        mask2 = F.interpolate(mask2, size=(h, w), mode='bilinear', align_corners=True)
        x = x * (self.alpha + (1 - self.alpha) / 2 * mask1 + (1 - self.alpha) / 2 * mask2)
        return x

    def forward(self, x, mask1, mask2):
        x = self.conv1(x)
        x = self.mask_x(x, mask1, mask2)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # 64, 256, 256
        x, _, _ = self.layer1(x, mask1, mask2)  # 64, 256, 256
        x, _, _ = self.layer2(x, mask1, mask2)  # 128, 128, 128
        x, _, _ = self.layer3(x, mask1, mask2)  # 256, 64, 64
        x, _, _ = self.layer4(x, mask1, mask2)  # 512, 32, 32

        x = self.avgpool_w(x)
        x = x.view(x.size(0), -1)
        x = self.fc_w(x)

        return x


class make_layer_alpha_mask(nn.Module):

    def __init__(self, inplanes, block, block_wam, planes, block_num, stride=1, block_type='default', windows_num=3,
                 initial_method="Uniform", k=0.8, alpha=0.5, if_mask=False):
        super().__init__()
        downsample = None
        self.inplanes = inplanes
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        self.layer = []
        self.block_num = block_num
        if block_type == 'default' or block_type == 'wam_half':
            self.layer.append(block(self.inplanes, planes, stride, downsample, alpha=alpha, if_mask=if_mask))
        elif block_type == 'wam_whole':
            self.layer.append(block_wam(self.inplanes, planes, windows_num=windows_num, initial_method=initial_method
                                        , block='basic_block', k=k, stride=stride, downsample=downsample, alpha=alpha,
                                        if_mask=if_mask))

        self.inplanes = planes
        for _ in range(1, block_num):
            if block_type == 'default':
                self.layer.append(block(self.inplanes, planes, alpha=alpha, if_mask=if_mask))
            elif block_type == 'wam_half' or block_type == 'wam_whole':
                self.layer.append(
                    block_wam(self.inplanes, planes, windows_num=windows_num, initial_method=initial_method
                              , block='basic_block', k=k, stride=stride, down_sample=downsample
                              , alpha=alpha, if_mask=if_mask))
        self.layer = nn.ModuleList(self.layer)

    def forward(self, x, mask1, mask2):
        for i in range(self.block_num):
            x, _, _ = self.layer[i](x, mask1, mask2)
        return x, mask1, mask2


def feature_alpha(x, feature_w, feature_b):
    x = 0.5 * x + (x * feature_w + feature_b) * 0.5
    return x


class ResNet_wam_feature(nn.Module):

    def __init__(self, block, block_wam, layers, input_channel=3, num_classes=1000, zero_init_residual=False,
                 windows_num=3, initial_method="Uniform", k=0.8, layer_num=1):
        super(ResNet_wam_feature, self).__init__()
        self.inplanes = 64
        # 3 * 1024 * 1024
        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        # 64 * 512 * 512
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 64 * 256 * 256
        self.layer1 = self._make_layer(block, block_wam, planes=64, block_num=layers[0], stride=1,
                                       block_type='wam_half', windows_num=windows_num, initial_method=initial_method,
                                       k=k)
        # 64 * 256 * 256
        if layer_num >= 2:
            self.layer2 = self._make_layer(block, block_wam, planes=128, block_num=layers[1], stride=2,
                                           block_type='wam_half', windows_num=windows_num,
                                           initial_method=initial_method,
                                           k=k)
        else:
            self.layer2 = self._make_layer(block, block_wam, planes=128, block_num=layers[1], stride=2)
        # 128 *
        if layer_num >= 3:
            self.layer3 = self._make_layer(block, block_wam, planes=256, block_num=layers[2], stride=2,
                                           block_type='wam_half', windows_num=windows_num,
                                           initial_method=initial_method,
                                           k=k)
        else:
            self.layer3 = self._make_layer(block, block_wam, planes=256, block_num=layers[2], stride=2)

        if layer_num >= 4:
            self.layer4 = self._make_layer(block, block_wam, planes=512, block_num=layers[3], stride=2,
                                           block_type='wam_half', windows_num=windows_num,
                                           initial_method=initial_method,
                                           k=k)
        else:
            self.layer4 = self._make_layer(block, block_wam, planes=512, block_num=layers[3], stride=2)

        self.featureCNN_w1 = feature_match(64, 64, downsample=1)
        self.featureCNN_w2 = feature_match(64, 128, downsample=2)
        self.featureCNN_w3 = feature_match(64, 256, downsample=4)
        self.featureCNN_w4 = feature_match(64, 512, downsample=8)

        self.featureCNN_b1 = feature_match(64, 64, downsample=1)
        self.featureCNN_b2 = feature_match(64, 128, downsample=2)
        self.featureCNN_b3 = feature_match(64, 256, downsample=4)
        self.featureCNN_b4 = feature_match(64, 512, downsample=8)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, block_wam, planes, block_num,
                    stride=1, block_type='default', windows_num=3, initial_method="Uniform", k=0.8):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if block_type == 'default' or block_type == 'wam_half':
            layers.append(block(self.inplanes, planes, stride, downsample))
        elif block_type == 'wam_whole':
            layers.append(block_wam(self.inplanes, planes, windows_num=windows_num, initial_method=initial_method
                                    , block='basic_block', k=k, stride=stride, downsample=downsample))

        self.inplanes = planes * block.expansion
        for _ in range(1, block_num):
            if block_type == 'default':
                layers.append(block(self.inplanes, planes))
            elif block_type == 'wam_half' or block_type == 'wam_whole':
                layers.append(block_wam(self.inplanes, planes, windows_num=windows_num, initial_method=initial_method
                                        , block='basic_block', k=k, stride=stride, down_sample=downsample))

        return nn.Sequential(*layers)

    def forward(self, x, feature):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        feature_w1 = self.featureCNN_w1(feature)
        feature_b1 = self.featureCNN_b1(feature)
        x = feature_alpha(x, feature_w1, feature_b1)

        x = self.layer2(x)
        feature_w2 = self.featureCNN_w2(feature)
        feature_b2 = self.featureCNN_b2(feature)
        x = feature_alpha(x, feature_w2, feature_b2)

        x = self.layer3(x)
        feature_w3 = self.featureCNN_w3(feature)
        feature_b3 = self.featureCNN_b3(feature)
        x = feature_alpha(x, feature_w3, feature_b3)

        x = self.layer4(x)
        feature_w4 = self.featureCNN_w4(feature)
        feature_b4 = self.featureCNN_b4(feature)
        x = feature_alpha(x, feature_w4, feature_b4)


        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34_wam_alpha_mask(input_channel=3, num_class=2, pretrained=False, windows_num=3, initial_method="Uniform", k=0.8
                            , layer_num=1, alpha=0.5, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        :param layer_num:
        :param alpha:
        :param input_channel:
        :param pretrained:
        :param k:
        :param initial_method:
        :param windows_num:
    """
    model = ResNet_alpha_mask(BasicBlock_alpha_mask, WamBlock_alpha_mask, [3, 4, 6, 3], input_channel=input_channel,
                              num_classes=num_class,
                              windows_num=windows_num,
                              initial_method=initial_method, k=k, layer_num=layer_num, alpha=alpha, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet34_wam(input_channel=3, pretrained=False, windows_num=3, initial_method="Uniform", k=0.8, layer_num=1,
                 **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        :param input_channel:
        :param pretrained:
        :param k:
        :param initial_method:
        :param windows_num:
    """
    model = ResNet_wam(BasicBlock, WamBlock, [3, 4, 6, 3], input_channel=input_channel, windows_num=windows_num,
                       initial_method=initial_method, k=k, layer_num=layer_num, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet34_wam_feature(input_channel=3, pretrained=False, windows_num=3, initial_method="Uniform", k=0.8, layer_num=1,
                         **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        :param input_channel:
        :param pretrained:
        :param k:
        :param initial_method:
        :param windows_num:
    """
    model = ResNet_wam_feature(BasicBlock, WamBlock, [3, 4, 6, 3], input_channel=input_channel, windows_num=windows_num,
                               initial_method=initial_method, k=k, layer_num=layer_num, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
