import torch.nn as nn
import torch
import cv2
import numpy as np
class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):
	assert np.all(np.equal(input1,input2) == 1)
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True)
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True)
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.mm(input2_l2.t()).pow(2)))

        return diff_loss
loss = DiffLoss()
root = '/home/kevin/Downloads/CityScapes/leftImg8bit/val/frankfurt/'
in1 = cv2.imread(root+'frankfurt_000001_011715_leftImg8bit.png').astype(np.float64)
in2 = cv2.imread(root+'frankfurt_000001_011835_leftImg8bit.png').astype(np.float64)
in1 = torch.FloatTensor(in1)
in2 = torch.FloatTensor(in2)

assert np.all(np.equal(in1, in1)==1), 'no'
l = loss(in1, in2)
print l
