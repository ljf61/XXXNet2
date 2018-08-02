#coding=UTF-8

from net_parts import *
import torch

class VONet(nn.Module):
    """docstring for VONet"""
    def __init__(self):
        super(VONet, self).__init__()
        self.inc = initial_conv(3,4)
        self.conv = initial_conv(256,256)
        self.down1 = down(3,16,32)
        self.down2 = down(3,32,64)
        self.down3 = down(3,64,128)
        self.down4 = down(3,128,256)
        self.down5 = down(3,256,256)
        self.up1 = only_up(3,256,256)
        self.up2 = only_up(3,256,128)
        self.up3 = only_up(3,128,64)
        self.up4 = up(3,128,32)
        self.up5 = up(3,64,16)
        self.up6 = only_up(3,16,16)
        self.depthR = out_conv(3,16,1)
        self.poseR = nn.Sequential(
            fc_layer(256*7*18,2048),
            fc_layer(2048,1024),
            fc_layer(1024,256),
            nn.Linear(256,2))
        self.atten_r = Atten_block_r(400)
        self.atten_c = Atten_block_c(400)
    
    def forward(self, xl, xr):
		xl = self.inc(xl)
		xr = self.inc(xr)

		xr_r = self.atten_r(xl, xr, xr)
		xl_r = self.atten_r(xr, xl, xl)
		xl_c = self.atten_c(xr, xl, xl)
		xr_c = self.atten_c(xl, xr, xr)
		x = torch.cat([xl_c,xr_c,xl_r,xr_r], dim=1)

		x1 = self.down1(x)
		x2 = self.down2(x1)
		x3 = self.down3(x2)
		x4 = self.down4(x3)
		x5 = self.down5(x4)

		x6 = self.conv(x5)
		#print(x5.size())
		featureVec = x6.view((-1,256*7*18))
		depth = self.up1(x6)
		depth = self.up2(depth)
		depth = self.up3(depth)
		depth = self.up4(depth,x2)
		depth = self.up5(depth,x1)
		depth = self.up6(depth)
		#print(depth.size())
		depth = self.depthR(depth)
		#print(depth.size())
		pose = self.poseR(featureVec)
		return depth,pose

