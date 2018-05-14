# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi he, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.test import test_net, im_detect
from model.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import argparse
import pprint
import cv2
import time, os, sys

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.mobilenet_v1 import mobilenetv1
import torch.nn as nn


import torch

if __name__ == '__main__':
  # if has model, get the name from it
  # if does not, then just use the initialization weights

  imdb = get_imdb('cityscapes_val')
  imdb.competition_mode(True)

  net = vgg16()

  # load model
  net.create_architecture(imdb.num_classes, tag='default',
                          anchor_scales=[4,8,16,32],
                          anchor_ratios=cfg.ANCHOR_RATIOS)

  net.eval()
  net.cuda()
  

  net.load_state_dict(torch.load('/home/disk1/DA/pytorch-faster-rcnn/output/vgg16/KITTI_train/default/vgg16_faster_rcnn_iter_490000.pth'))
  im = cv2.imread('/home/kevin/Downloads/CityScapes/gtFine/test/berlin/berlin_000000_000019_gtFine_color.png')
  print('first pass')
  scores, boxes, fc7, net_conv = im_detect(net, im)


  net.load_state_dict(torch.load('/home/disk1/DA/pytorch-faster-rcnn/output/vgg16/KITTI_train/_adapt/comb_const/vgg16_faster_rcnn_iter_30000.pth'))
  print('second pass')
  scores, boxes, fc7_adapt, net_conv_adapt = im_detect(net, im)

  print('compute loss')
  #loss = nn.L1Loss()
  loss = nn.MSELoss()
  print(fc7.size())
  print(loss(fc7, fc7_adapt))
  print(net_conv.size())
  print(loss(net_conv, net_conv_adapt))