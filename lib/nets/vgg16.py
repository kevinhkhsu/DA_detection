# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nets.network import Network
from model.config import cfg

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models

class vgg16(Network):
  def __init__(self):
    Network.__init__(self)
    self._feat_stride = [16, ]
    self._feat_compress = [1. / float(self._feat_stride[0]), ]
    self._net_conv_channels = 512
    self._fc7_channels = 4096

  def _init_head_tail(self):
    self.vgg = models.vgg16()
    # Remove fc8
    self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier._modules.values())[:-1])

    # Fix the layers before conv3:
    #for layer in range(10):
    #  for p in self.vgg.features[layer].parameters(): p.requires_grad = False

    # self.vgg.features._modules['28'] = nn.Conv2d(512, 1024, [3, 3], padding=1) #for feature_separate

    # not using the last maxpool layer
    self._layers['head'] = nn.Sequential(*list(self.vgg.features._modules.values())[:-1])

    ##
    # self.vgg2 = models.vgg16()
    # self._layers['head_2'] = nn.Sequential(*list(self.vgg2.features._modules.values())[:-1])

  def _image_to_head(self):
    net_conv = self._layers['head'](self._image)
    self._act_summaries['conv'] = net_conv
     
    return net_conv

  # def _image_to_head_branch(self):
  #   net_conv2 = self._layers['head_2'](self._image)
  
  #   return net_conv2

  def _head_to_tail(self, pool5):
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.vgg.classifier(pool5_flat)

    return fc7

  def load_pretrained_cnn(self, state_dict):
    #load from previous network weight
    netDict = self.state_dict()
    stateDict = {k: v for k, v in state_dict.items() if k in netDict}
    
    #print('load pretrained:', stateDict.keys())
    netDict.update(stateDict)
    nn.Module.load_state_dict(self, netDict)
    self.vgg.load_state_dict({k.replace('vgg.', ''):v for k,v in state_dict.items() if k.replace('vgg.', '') in self.vgg.state_dict()}) #loading pretrained vgg.pth

