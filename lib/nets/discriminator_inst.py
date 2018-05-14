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

class FCDiscriminator_inst(nn.Module):

	def __init__(self, in_channel, ndf = 4096):
		super(FCDiscriminator_inst, self).__init__()

		self.fc1 = nn.Linear(in_channel, ndf)
		self.fc2 = nn.Linear(ndf, ndf)
		self.fc3 = nn.Linear(ndf, ndf)
		self.classifier = nn.Linear(ndf, 1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		self.dropout = nn.Dropout()


	def forward(self, x):
		x = x.view(x.size()[0], -1)
		x = self.fc1(x)
		x = self.leaky_relu(x)
		# x = self.dropout(x)
		x = self.fc2(x)
		x = self.leaky_relu(x)
		# x = self.dropout(x)
		x = self.fc3(x)
		x = self.leaky_relu(x)
		# x = self.dropout(x)
		x = self.classifier(x)
                
		return x