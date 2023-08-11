# -*- coding: utf-8 -*-
# @Time    : 2023/3/2 14:53
# @Author  : wth
# @FileName: fusion_image_net.py
# @Software: PyCharm
from functools import partial
import torch
from torch import nn

from basic_net import SELayer, Attention, Mlp, DropPath

__all__ = ['fusion_img_net', 'fusion_img_net_ablation_cnn', 'fusion_img_net_ablation_trans', 'fusion_img_net',
           'fusion_img_net_new_split_img', 'resnet_seg', 'fusion_img_net_cnn_weight']

from basic_net.wam_block import WamBlock


class FusionImgNetLast(nn.Module):
    def __init__(self, block, layers, num_classes=512, image_size=1024, windows_num=3,
                 layer_num=1, initial_method="Uniform", k=0.8, zero_init_residual=False, input_channel=3):
        super(FusionImgNetLast, self).__init__()
        self.ResNet_Module = ResNet_Module(block, layers, input_channel)
        self.Transformer_Module = TransformerModuleNoShape(image_size, input_channel=input_channel, embed_dim=128)
        self.SE_Fusion_Module = SeFusionModule(mix_in_channel=320)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(640, num_classes)

    def forward(self, x):
        trans_x = self.Transformer_Module(x)
        x = self.ResNet_Module(x)
        x = self.avg_pool(x)
        x = torch.cat((x, trans_x), dim=1)
        x = self.SE_Fusion_Module(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class FusionImgNetCnnWeight(nn.Module):
    def __init__(self, block, layers, num_classes=512, image_size=1024, windows_num=3,
                 layer_num=1, initial_method="Uniform", k=0.8, zero_init_residual=False, input_channel=3):
        super(FusionImgNetCnnWeight, self).__init__()
        self.ResNet_Module = ResNet_Module(block, layers, input_channel)
        self.Transformer_Module = Transformer_Module_ablation(image_size, input_channel=input_channel, embed_dim=128)
        self.SE_Fusion_Module = SeFusionModule(mix_in_channel=320)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(640, num_classes)

    def forward(self, x):
        trans_x = self.Transformer_Module(x)
        x = self.ResNet_Module(x)
        x = self.avg_pool(x)
        x = torch.cat((x, trans_x), dim=1)
        x = self.SE_Fusion_Module(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class FusionImgNetAblationTrans(nn.Module):
    def __init__(self, block, layers, num_classes=512, image_size=1024, windows_num=3,
                 layer_num=1, initial_method="Uniform", k=0.8, zero_init_residual=False, input_channel=3,
                 ablation='trans'):
        super(FusionImgNetAblationTrans, self).__init__()
        self.ResNet_Module = ResNet_Module(block, layers, input_channel)
        self.Transformer_Module = TransformerModuleNoShape(image_size, input_channel=input_channel, embed_dim=128)
        self.SE_Fusion_Module = SeFusionModule(mix_in_channel=64)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.Transformer_Module(x)
        x = self.avg_pool(x)
        x = self.SE_Fusion_Module(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class SeFusionModule(nn.Module):
    def __init__(self, mix_in_channel=512):
        super(SeFusionModule, self).__init__()
        self.mix_in_channel = mix_in_channel
        self.se = SELayer(channel=2 * self.mix_in_channel, reduction=16)

    def forward(self, x):
        x = self.se(x)
        return x


class TransformerModuleNoShape(nn.Module):
    def __init__(self, image_size=1024, input_channel=3, embed_dim=512):
        super(TransformerModuleNoShape, self).__init__()

        self.patch_size = 32
        self.embed_dim = embed_dim
        self.num_heads = 8
        self.num_tokens = 0
        self.feature_channel = 16
        self.feature_pad_channel = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=self.feature_channel - input_channel, kernel_size=5,
                      stride=1, padding=2),
            nn.BatchNorm2d(self.feature_channel - input_channel),
            nn.ReLU(inplace=True)
        )
        self.feature_block = BasicBlock(self.feature_channel - input_channel, self.feature_channel - input_channel,
                                        stride=1)
        self.patch_embed = PatchEmbed(img_size=image_size, patch_size=self.patch_size, in_c=self.feature_channel,
                                      embed_dim=self.embed_dim)
        self.norm_layer = partial(nn.LayerNorm, eps=1e-6)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, self.embed_dim))
        self.pos_drop = nn.Dropout(p=0.1)
        self.transformer_block1 = TransformerBlock(dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=2,
                                                   qkv_bias=False, qk_scale=None,
                                                   drop_ratio=0.1, attn_drop_ratio=0.1, drop_path_ratio=0.1,
                                                   norm_layer=self.norm_layer, act_layer=nn.GELU)
        self.transformer_block2 = TransformerBlock(dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=2,
                                                   qkv_bias=False, qk_scale=None,
                                                   drop_ratio=0.1, attn_drop_ratio=0.1, drop_path_ratio=0.2,
                                                   norm_layer=self.norm_layer, act_layer=nn.GELU)
        # self.reshape_transformer = ReshapeTransform(input_size=[image_size, image_size],
        #                                             patch_size=[self.patch_size, self.patch_size])

    def forward(self, x):
        # x : [B, C, H, W]
        feature = self.feature_pad_channel(x)
        feature = self.feature_block(feature)
        x = torch.cat((x, feature), dim=1)
        x = self.patch_embed(x)
        # [B, C, H, W] -> [B, HW/patch_size^2, dim]
        x = self.pos_drop(x + self.pos_embed)
        x = self.transformer_block1(x)
        x = self.transformer_block2(x)
        x = x[:, 0, :]
        x = torch.unsqueeze(x, dim=2)
        x = torch.unsqueeze(x, dim=3)
        # x = x.permute(0, 2, 1, 3)
        # [B, dim, 1, 1]
        return x


class TransformerModule(nn.Module):
    def __init__(self, image_size=1024, input_channel=3, embed_dim=512):
        super(TransformerModule, self).__init__()

        self.patch_size = 32
        self.embed_dim = embed_dim
        self.num_heads = 8
        self.num_tokens = 0
        self.feature_channel = 16
        self.feature_pad_channel = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=self.feature_channel - input_channel, kernel_size=5,
                      stride=1, padding=2),
            nn.BatchNorm2d(self.feature_channel - input_channel),
            nn.ReLU(inplace=True)
        )
        self.feature_block = BasicBlock(self.feature_channel - input_channel, self.feature_channel - input_channel,
                                        stride=1)
        self.patch_embed = PatchEmbed(img_size=image_size, patch_size=self.patch_size, in_c=self.feature_channel,
                                      embed_dim=self.embed_dim)
        self.norm_layer = partial(nn.LayerNorm, eps=1e-6)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, self.embed_dim))
        self.pos_drop = nn.Dropout(p=0.1)
        self.transformer_block1 = TransformerBlock(dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=2,
                                                   qkv_bias=False, qk_scale=None,
                                                   drop_ratio=0.1, attn_drop_ratio=0.1, drop_path_ratio=0.1,
                                                   norm_layer=self.norm_layer, act_layer=nn.GELU)
        self.transformer_block2 = TransformerBlock(dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=2,
                                                   qkv_bias=False, qk_scale=None,
                                                   drop_ratio=0.1, attn_drop_ratio=0.1, drop_path_ratio=0.2,
                                                   norm_layer=self.norm_layer, act_layer=nn.GELU)
        self.reshape_transformer = ReshapeTransform(input_size=[image_size, image_size],
                                                    patch_size=[self.patch_size, self.patch_size])

    def forward(self, x):
        # x : [B, C, H, W]
        feature = self.feature_pad_channel(x)
        feature = self.feature_block(feature)
        x = torch.cat((x, feature), dim=1)
        x = self.patch_embed(x)
        # [B, C, H, W] -> [B, HW/patch_size^2, dim]
        x = self.pos_drop(x + self.pos_embed)
        x = self.transformer_block1(x)
        x = self.transformer_block2(x)
        x = self.reshape_transformer(x)
        # [B, HW/patch_size^2, dim] -> [b, dim, H/patch_size, W/patch_size]
        return x


class Transformer_Module_ablation(nn.Module):

    def __init__(self, image_size=1024, input_channel=3, embed_dim=512):
        super(Transformer_Module_ablation, self).__init__()

        self.patch_size = 32
        self.embed_dim = embed_dim
        self.num_heads = 8
        self.num_tokens = 0
        self.feature_channel = 16
        self.feature_pad_channel = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=self.feature_channel - input_channel, kernel_size=5,
                      stride=1, padding=2),
            nn.BatchNorm2d(self.feature_channel - input_channel),
            nn.ReLU(inplace=True)
        )
        self.feature_block = BasicBlock(self.feature_channel - input_channel, self.feature_channel - input_channel,
                                        stride=1)
        self.patch_embed = PatchEmbed(img_size=image_size, patch_size=self.patch_size, in_c=self.feature_channel,
                                      embed_dim=self.embed_dim)
        self.norm_layer = partial(nn.LayerNorm, eps=1e-6)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, self.embed_dim))
        self.pos_drop = nn.Dropout(p=0.1)
        # self.transformer_block1 = TransformerBlock(dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=2,
        #                                            qkv_bias=False, qk_scale=None,
        #                                            drop_ratio=0.1, attn_drop_ratio=0.1, drop_path_ratio=0.1,
        #                                            norm_layer=self.norm_layer, act_layer=nn.GELU)
        # self.transformer_block2 = TransformerBlock(dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=2,
        #                                            qkv_bias=False, qk_scale=None,
        #                                            drop_ratio=0.1, attn_drop_ratio=0.1, drop_path_ratio=0.2,
        #                                            norm_layer=self.norm_layer, act_layer=nn.GELU)
        self.reshape_transformer = ReshapeTransform(input_size=[image_size, image_size],
                                                    patch_size=[self.patch_size, self.patch_size])

    def forward(self, x):
        # x : [B, C, H, W]
        feature = self.feature_pad_channel(x)
        feature = self.feature_block(feature)
        x = torch.cat((x, feature), dim=1)
        x = self.patch_embed(x)
        # [B, C, H, W] -> [B, HW/patch_size^2, dim]
        x = self.pos_drop(x + self.pos_embed)
        # x = self.transformer_block1(x)
        # x = self.transformer_block2(x)
        x = x[:, 0, :]
        x = torch.unsqueeze(x, dim=2)
        x = torch.unsqueeze(x, dim=3)
        return x


class ResNet_Module(nn.Module):

    def __init__(self, block, layers, input_channel=3, zero_init_residual=False):
        super(ResNet_Module, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.wamblock = WamBlock(input_channel, input_channel, windows_num=3)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

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

    def _make_layer(self, block, planes, blocks, stride=1):
        down_sample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            down_sample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, down_sample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

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

        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, down_sample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
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


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=512, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # proj:[B, C, H, W] -> [B, dim, H/patch_size, W/patch_size]
        # flatten: [B, dim, H/patch_size, W/patch_size] -> [B, dim, HW/patch_size^2]
        # transpose: [B, dim, HW/patch_size^2] -> [B, HW/patch_size^2, dim]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(TransformerBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ReshapeTransform(nn.Module):
    def __init__(self, input_size, patch_size):
        super(ReshapeTransform, self).__init__()
        self.h = input_size[0] // patch_size[0]
        self.w = input_size[1] // patch_size[1]

    def forward(self, x):
        # remove cls token and reshape
        # [batch_size, num_tokens, token_dim]
        # [b, HW/patch_size^2, dim] -> [b, H/patch_size, W/patch_size, dim]
        result = x.reshape(x.size(0), self.h, self.w, x.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        # [b, H/patch_size, W/patch_size, dim] -> [b, dim, H/patch_size, W/patch_size]
        result = result.permute(0, 3, 1, 2)
        return result


# CNN Ablation
def fusion_img_net_cnn_weight(windows_num=3, initial_method="Uniform", k=0.8, layer_num=1, num_classes=2,
                              input_channel=3):
    model = FusionImgNetCnnWeight(BasicBlock, [5, 5, 5, 6], num_classes=num_classes, image_size=1024,
                                  windows_num=windows_num, initial_method=initial_method, layer_num=layer_num,
                                  input_channel=input_channel)
    return model


# Trans Ablation
def fusion_img_net_ablation_trans(windows_num=3, initial_method="Uniform", k=0.8, layer_num=1, num_classes=2,
                                  input_channel=3):
    model = FusionImgNetAblationTrans(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, image_size=1024,
                                      windows_num=windows_num, initial_method=initial_method, layer_num=layer_num,
                                      input_channel=input_channel, ablation='trans')
    return model


# Trans MUF
def fusion_img_net(windows_num=3, initial_method="Uniform", k=0.8, layer_num=1, num_classes=2, input_channel=3):
    model = FusionImgNetLast(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, image_size=1024,
                             windows_num=windows_num, initial_method=initial_method, layer_num=layer_num,
                             input_channel=input_channel)
    return model
