from ..builder import BACKBONES
from .resnet3d_slowonly import ResNet3dSlowOnly
from ...utils import cache_checkpoint, get_root_logger

import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint

@BACKBONES.register_module()
class Visece_Video(nn.Module):
    def __init__(self, res_out_channels, output_channels, pretrained=None, **kwargs):
        super().__init__()
        self.resnet = ResNet3dSlowOnly(**kwargs)
        self.pool2d = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(res_out_channels, output_channels)
        self.pretrained = pretrained

    def forward(self, x):
        x = self.resnet(x) # [B, C, T, H, W]
        B, C, T, H, W = x.shape
        x = x.reshape((1, -1,)+x.shape[3:])
        x = self.pool2d(x) # [B, CT, 1, 1]
        x = x.reshape(B, C, T) # [B, C, T]
        x = x.permute(0, 2, 1) # [B, T, C]
        x = self.linear(x) # [B, T, Cout]
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')
            self.pretrained = cache_checkpoint(self.pretrained)
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

@BACKBONES.register_module()
class Visece_Video_PACL(nn.Module):
    def __init__(self, res_out_channels, pretrained=None, **kwargs):
        super().__init__()
        self.resnet = ResNet3dSlowOnly(**kwargs)
        self.pool2d = nn.AdaptiveAvgPool2d(1)
        self.pretrained = pretrained

    def forward(self, x):
        x = self.resnet(x) # [B, C, T, H, W]
        B, C, T, H, W = x.shape
        x = x.reshape((1, -1,)+x.shape[3:])
        x = self.pool2d(x) # [B, CT, 1, 1]
        x = x.reshape(B, C, T) # [B, C, T]
        x = x.permute(0, 2, 1) # [B, T, C=res_out_channels]
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')
            self.pretrained = cache_checkpoint(self.pretrained)
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)