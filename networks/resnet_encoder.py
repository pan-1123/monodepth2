# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()

class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    Adapted from https://github.com/nianticlabs/monodepth2/edit/master/networks/resnet_encoder.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        # self.inplanes = 64
        # self.conv1 = nn.Conv2d(
        #    num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        in_ch, out_ch = 64,96
        self.deconv1 = nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=4, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # for m in self.modules():
            # if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, nn.BatchNorm2d):
                # nn.init.constant_(m.weight, 1)
                # nn.init.constant_(m.bias, 0)
          for m in self.modules():
              if isinstance(m, nn.ConvTranspose2d):
                  c1, c2, h, w = m.weight.data.size()
                  weight = get_upsample_filter(h)
                  m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                  if m.bias is not None:
                      m.bias.data.zero_()     
         
def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    # def forward(self, input_image):
        # self.features = []
        # x = (input_image - 0.45) / 0.225
        # x = self.encoder.conv1(x)
        # x = self.encoder.bn1(x)
        # self.features.append(self.encoder.relu(x))
        # self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        
    def forward(self) :
        self.features = []
        dct_y  = torch.from_numpy(np.random.randn(16, 64, 28, 28)).float()
        dc_cb = torch.from_numpy(np.random.randn(16, 64, 14, 14)).float()
        dc_cr = torch.from_numpy(np.random.randn(16, 64, 14, 14)).float()
        
        dct_cb = self.deconv1(dc_cb)
        dct_cr = self.deconv2(dc_cr)
        x = torch.cat((dct_y, dct_cb, dct_cr), 1)   
        self.features.append(self.encoder.layer1(x))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features
