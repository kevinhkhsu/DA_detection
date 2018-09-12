# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import cfg

class decoder(nn.Module):

	def __init__(self, num_classes):
		super(decoder, self).__init__()

		self.conv6_1 = nn.ConvTranspose2d(num_classes, 512, kernel_size=3, padding=1)
		self.conv6_2 = nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1)
		self.conv6_3 = nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1)
		self.unpool1 = nn.Upsample(scale_factor=2, mode='bilinear')

		self.conv7_1 = nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1)
		self.conv7_2 = nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1)
		self.conv7_3 = nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1)
		self.unpool2 = nn.Upsample(scale_factor=2, mode='bilinear')

		self.conv8_1 = nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1)
		self.conv8_2 = nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1)
		self.conv8_3 = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
		self.unpool3 = nn.Upsample(scale_factor=2, mode='bilinear')

		self.conv9_1 = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1)
		self.conv9_2 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
		self.unpool4 = nn.Upsample(scale_factor=2, mode='bilinear')

		self.conv10_1 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1)
		self.conv10_2 = nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1)

		# self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		self.relu = nn.ReLU(inplace=True)
		self.sigmoid = nn.Sigmoid()


	def forward(self, x):
		x = self.conv6_1(x)
		x = self.relu(x)
		x = self.conv6_2(x)
		x = self.relu(x)
		x = self.conv6_3(x)
		x = self.relu(x)
		x = self.unpool1(x)

		x = self.conv7_1(x)
		x = self.relu(x)
		x = self.conv7_2(x)
		x = self.relu(x)
		x = self.conv7_3(x)
		x = self.relu(x)
		x = self.unpool2(x)

		x = self.conv8_1(x)
		x = self.relu(x)
		x = self.conv8_2(x)
		x = self.relu(x)
		x = self.conv8_3(x)
		x = self.relu(x)
		x = self.unpool3(x)

		x = self.conv9_1(x)
		x = self.relu(x)
		x = self.conv9_2(x)
		x = self.relu(x)
		x = self.unpool4(x)

		x = self.conv10_1(x)
		x = self.relu(x)
		x = self.conv10_2(x)
		x = self.sigmoid(x)

		return x