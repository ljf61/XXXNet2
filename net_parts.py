#coding=UTF-8

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class initial_conv(nn.Module):
	def __init__(self, in_ch, out_ch):
	    super(initial_conv, self).__init__()
	    self.conv = nn.Sequential(
			nn.Conv2d(in_ch, in_ch, 3, padding=1,bias=True),
			nn.BatchNorm2d(in_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_ch, in_ch, 3, padding=1,bias=True),
			nn.BatchNorm2d(in_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_ch, out_ch, 3, padding=1,bias=True),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True))

	def forward(self, x):
		x = self.conv(x)
		return x

class out_conv(nn.Module):
    def __init__(self, kernel_size, in_ch, out_ch):
        super(out_conv, self).__init__()
        self.conv = nn.Sequential(
	    nn.Conv2d(in_ch, out_ch, kernel_size, padding=1,bias=True))
		
    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, kernel_size, in_ch, out_ch):
        super(down, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=1,bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1,bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Dropout2d(p=0.2))

    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, kernel_size, in_ch, out_ch):
	    super(up, self).__init__()
	    self.conv = nn.Sequential(
	        nn.Conv2d(in_ch, out_ch, kernel_size, padding=1,bias=True),
	        nn.ReLU(inplace=True),
	        nn.Conv2d(out_ch, out_ch, 3, padding=1,bias=True),
	        nn.ReLU(inplace=True))
	    self.up = nn.Upsample(scale_factor=2)

    def forward(self,x1,x2):
    	x1 = self.up(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffY // 2, int((diffY+1) / 2),diffX // 2, int((diffX+1) / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class only_up(nn.Module):
	def __init__(self, kernel_size, in_ch, out_ch):
		super(only_up, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size, padding=1,bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, 3, padding=1,bias=True),
			nn.ReLU(inplace=True))
		self.up = nn.Upsample(scale_factor=2)

	def forward(self,x1):
		x = self.up(x1)
		x = self.conv(x)
		return x

class fc_layer(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(fc_layer, self).__init__()
		self.fc = nn.Sequential(
			nn.Dropout(p=0.2),
			nn.Linear(in_ch,out_ch,bias=True),
			nn.ReLU(inplace=True))
		    

	def forward(self,x):
		x = self.fc(x)
		return x

class Atten_block_r(nn.Module):
	def __init__(self, d_model):
		super(Atten_block_r, self).__init__()
		self.temper = np.power(d_model, 0.5)
		self.softmax = nn.Softmax2d()

	def forward(self, q, k, v):
		attention = torch.matmul(q, k.transpose(2,3))/self.temper
		attention = self.softmax(attention)
		output = torch.matmul(attention,v)
		return output	
		
class Atten_block_c(nn.Module):
	def __init__(self, d_model):
		super(Atten_block_c, self).__init__()
		self.temper = np.power(d_model, 0.5)
		self.softmax = nn.Softmax2d()

	def forward(self, q, k, v):
		attention = torch.matmul(q.transpose(2,3), k)/self.temper
		attention = self.softmax(attention)
		output = torch.matmul(attention,v.transpose(2,3))
		return output.transpose(2,3)
