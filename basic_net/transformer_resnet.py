from functools import partial

import torch
import torch.nn as nn

from basic_net.vit_model import Attention, DropPath, Mlp
from basic_net.se_module import SELayer

__all__ = ['transformer_resnet_loc3', 'transformer_resnet_loc4', 'transformer_resnet_wam_loc4',
           'transformer_resnet_wam_se_loc4']

from basic_net.wam_block import WamBlock


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


class TransformerResnet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, image_size=1024, layer_loc=4, zero_init_residual=False):
        super(TransformerResnet, self).__init__()
        self.inplanes = 64
        self.embed_dim = 512
        self.num_heads = 8
        self.patch_size = 16
        self.num_tokens = 0
        self.norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.layer_loc = layer_loc
        if layer_loc == 4:
            self.patch_size = 32
            self.embed_dim = 512
        elif layer_loc == 3:
            self.patch_size = 16
            self.embed_dim = 256
        else:
            raise ValueError('layer location just can set on 3 or 4')

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.patch_embed = PatchEmbed(img_size=image_size, patch_size=self.patch_size, in_c=3, embed_dim=self.embed_dim)
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
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward_transformer(self, x):
        # x : [B, C, H, W]
        x = self.patch_embed(x)
        # [B, C, H, W] -> [B, HW/patch_size^2, dim]
        x = self.pos_drop(x + self.pos_embed)
        x = self.transformer_block1(x)
        x = self.transformer_block2(x)
        x = self.reshape_transformer(x)
        # [B, HW/patch_size^2, dim] -> [b, dim, H/patch_size, W/patch_size]
        return x

    def forward(self, x):
        trans_x = self.forward_transformer(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.layer_loc == 3:
            x = x + trans_x
        x = self.layer4(x)
        if self.layer_loc == 4:
            x = x + trans_x

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class TransformerResnetWAM(nn.Module):

    def __init__(self, block, layers, num_classes=1000, image_size=1024, layer_loc=4, windows_num=3,
                 layer_num=1, initial_method="Uniform", k=0.8, zero_init_residual=False, input_channel=3):
        super(TransformerResnetWAM, self).__init__()
        self.inplanes = 64
        self.embed_dim = 512
        self.num_heads = 8
        self.patch_size = 16
        self.num_tokens = 0
        self.norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.layer_loc = layer_loc
        if layer_loc == 4:
            self.patch_size = 32
            self.embed_dim = 512
        elif layer_loc == 3:
            self.patch_size = 16
            self.embed_dim = 256
        else:
            raise ValueError('layer location just can set on 3 or 4')

        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        block_wam = WamBlock
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

        self.patch_embed = PatchEmbed(img_size=image_size, patch_size=self.patch_size, in_c=input_channel, embed_dim=self.embed_dim)
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
                                        , block='basic_block', k=k))

        return nn.Sequential(*layers)

    def forward_transformer(self, x):
        # x : [B, C, H, W]
        x = self.patch_embed(x)
        # [B, C, H, W] -> [B, HW/patch_size^2, dim]
        x = self.pos_drop(x + self.pos_embed)
        x = self.transformer_block1(x)
        x = self.transformer_block2(x)
        x = self.reshape_transformer(x)
        # [B, HW/patch_size^2, dim] -> [b, dim, H/patch_size, W/patch_size]
        return x

    def forward(self, x):
        trans_x = self.forward_transformer(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.layer_loc == 3:
            x = x + trans_x
        x = self.layer4(x)
        if self.layer_loc == 4:
            x = x + trans_x

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class TransformerResnetWamSE(nn.Module):

    def __init__(self, block, layers, num_classes=1000, image_size=1024, layer_loc=4, windows_num=3,
                 layer_num=1, initial_method="Uniform", k=0.8, zero_init_residual=False, input_channel=3):
        super(TransformerResnetWamSE, self).__init__()
        self.inplanes = 64
        self.embed_dim = 512
        self.num_heads = 8
        self.patch_size = 16
        self.num_tokens = 0
        self.norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.layer_loc = layer_loc
        if layer_loc == 4:
            self.patch_size = 32
            self.embed_dim = 512
        elif layer_loc == 3:
            self.patch_size = 16
            self.embed_dim = 256
        else:
            raise ValueError('layer location just can set on 3 or 4')

        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        block_wam = WamBlock
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

        self.feature_channel = 16
        self.feature_pad_channel = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=self.feature_channel - input_channel, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(self.feature_channel - input_channel),
            nn.ReLU(inplace=True)
        )
        self.feature_block = WamBlock(in_channel=self.feature_channel - input_channel, out_channel=self.feature_channel - input_channel,
                                      windows_num=3
                                      , block='basic_block', k=k, stride=1)

        self.patch_embed = PatchEmbed(img_size=image_size, patch_size=self.patch_size, in_c=self.feature_channel,
                                      embed_dim=self.embed_dim)
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
        if self.layer_loc == 3:
            self.mix_in_channel = 256
        elif self.layer_loc == 4:
            self.mix_in_channel = 512
        self.se = SELayer(channel=2 * self.mix_in_channel, reduction=16)
        self.mix_block = nn.Sequential(
            nn.Conv2d(2 * self.mix_in_channel, self.mix_in_channel, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(self.mix_in_channel),
            nn.ReLU(inplace=True),
            WamBlock(in_channel=self.mix_in_channel, out_channel=self.mix_in_channel, windows_num=3,
                     block='basic_block', k=k, stride=1)
        )

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
                                        , block='basic_block', k=k))

        return nn.Sequential(*layers)

    def forward_transformer(self, x):
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

    def forward(self, x):
        trans_x = self.forward_transformer(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.layer_loc == 3:
            x = torch.cat((x, trans_x), dim=1)
            x = self.se(x)
            x = self.mix_block(x)
        x = self.layer4(x)
        if self.layer_loc == 4:
            x = torch.cat((x, trans_x), dim=1)
            x = self.se(x)
            x = self.mix_block(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def transformer_resnet_loc4(num_classes=2):
    model = TransformerResnet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, layer_loc=4, image_size=1024)
    return model


def transformer_resnet_loc3(num_classes=2):
    model = TransformerResnet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, layer_loc=3, image_size=1024)
    return model


def transformer_resnet_wam_loc4(windows_num=3, initial_method="Uniform", k=0.8, layer_num=1, num_classes=2, input_channel=3):
    model = TransformerResnetWAM(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, layer_loc=4, image_size=1024,
                                 windows_num=windows_num, initial_method=initial_method, layer_num=layer_num, input_channel=input_channel)
    return model


def transformer_resnet_wam_se_loc4(windows_num=3, initial_method="Uniform", k=0.8, layer_num=1, num_classes=2, input_channel=3):
    model = TransformerResnetWamSE(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, layer_loc=4, image_size=1024,
                                   windows_num=windows_num, initial_method=initial_method, layer_num=layer_num, input_channel=input_channel)
    return model
