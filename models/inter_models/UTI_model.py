"""
Code borrows heavily from
Xiangyu Xu, Siyao Li, Wenxiu Sun, Qian Yin, and Ming-Hsuan Yang, "Quadratic Video Interpolation", NeurIPS 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .flow_reversal import FlowReversal
from .UNet2 import UNet2 as UNet
from .PWCNetnew import PWCNet
from .acceleration_UTI import AcFusionLayer as Acc_UTI
from .refine_S2 import RefineS2
import sys


def backwarp(img, flow):
    _, _, H, W = img.size()

    u = flow[:, 0, :, :]
    v = flow[:, 1, :, :]

    gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))

    gridX = torch.tensor(gridX, requires_grad=False,).cuda()
    gridY = torch.tensor(gridY, requires_grad=False,).cuda()
    x = gridX.unsqueeze(0).expand_as(u).float() + u
    y = gridY.unsqueeze(0).expand_as(v).float() + v
    # range -1 to 1
    x = 2*(x/W - 0.5)
    y = 2*(y/H - 0.5)
    # stacking X and Y
    grid = torch.stack((x,y), dim=3)
    # Sample pixels using bilinear interpolation.
    imgOut = torch.nn.functional.grid_sample(img, grid)

    return imgOut


class SmallMaskNet(nn.Module):
    """A three-layer network for predicting mask"""
    def __init__(self, input, output):
        super(SmallMaskNet, self).__init__()
        self.conv1 = nn.Conv2d(input, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, output, 3, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        x = self.conv3(x)
        return x


class UTI_esti(nn.Module):
    """ Uncertain time interval estimation"""
    def __init__(self,path='./utils/network-default.pytorch'):
        super(UTI_esti,self).__init__()
        self.flownet = PWCNet()
        self.refineS2 = RefineS2()
        self.refineS2.load_state_dict(torch.load('./pretrain_models/refine_net.pth'))
        self.flownet.load_state_dict(torch.load(path))

    def forward(self, I0,I1,I2,I3):
        F10 = self.flownet(I1, I0).float()
        F12 = self.flownet(I1, I2).float()
        F21 = self.flownet(I2, I1).float()
        F20 = self.flownet(I2, I0).float()
        
        F13 = self.flownet(I1, I3).float()
        F23 = self.flownet(I2, I3).float()
        
        F1_23 = F13 - F12
        F2_10 = F20 - F21

        # refine S23
        F1_23_refine = self.refineS2(F10, F12, F1_23)
        F2_10_refine = self.refineS2(F23, F21, F2_10)
        
        # forward lambda t1/t0
        S1_1 = torch.sqrt(F12[:,0,:,:]**2 + F12[:,1,:,:]**2)
        F1_02 = F1_23_refine - F10
        S1_02 = torch.sqrt(F1_02[:,0,:,:]**2 + F1_02[:,1,:,:]**2)
        lambda_f = 2*(S1_1) / S1_02
        r_mean = lambda_f.mean()
        print('lambda:%.02f'%(r_mean.item()))

        # reverse lambda t1/t0
        S2_1 = torch.sqrt(F21[:,0,:,:]**2 + F21[:,1,:,:]**2)
        S2_02 = F2_10_refine - F23
        S2_02 = torch.sqrt(S2_02[:,0,:,:]**2 + S2_02[:,1,:,:]**2)
        lambda_b = 2*(S2_1) / S2_02

        flow = {'F10':F10,'F12':F12,'F1_23':F1_23_refine,'F23':F23,'F21':F21,'F2_10':F2_10_refine}
        lambda_time_interval = {'forward':lambda_f,'backward':lambda_b}
        return flow, lambda_time_interval



class UTI_interp(nn.Module):
    """Quadratic Video Interpolation refine S1""" 
    def __init__(self):
        super(UTI_interp, self).__init__()

        self.flownet = PWCNet()
        self.acc = Acc_UTI()
        self.fwarp = FlowReversal()
        self.refinenet = UNet(20, 8)
        self.masknet = SmallMaskNet(38, 1)

        
    def forward(self, I1, I2, flow,lambda_t, index_t):

        F1ta = self.acc(flow['F10'], flow['F12'], flow['F1_23'], lambda_t['forward'], index_t)
        F2ta = self.acc(flow['F23'], flow['F21'], flow['F2_10'], lambda_t['backward'], 1-index_t)    
        F1t = F1ta
        F2t = F2ta


        # Flow Reversal
        Ft1, norm1 = self.fwarp(F1t, F1t)
        Ft1 = -Ft1
        Ft2, norm2 = self.fwarp(F2t, F2t)
        Ft2 = -Ft2

        Ft1[norm1 > 0] = Ft1[norm1 > 0]/norm1[norm1>0].clone()
        Ft2[norm2 > 0] = Ft2[norm2 > 0]/norm2[norm2>0].clone()


        I1t = backwarp(I1, Ft1)
        I2t = backwarp(I2, Ft2)

        output, feature = self.refinenet(torch.cat([I1, I2, I1t, I2t, flow['F12'], flow['F21'], Ft1, Ft2], dim=1))

        # Adaptive filtering
        Ft1r = backwarp(Ft1, 10*torch.tanh(output[:, 4:6])) + output[:, :2]
        Ft2r = backwarp(Ft2, 10*torch.tanh(output[:, 6:8])) + output[:, 2:4]

        I1tf = backwarp(I1, Ft1r)
        I2tf = backwarp(I2, Ft2r)

        M = torch.sigmoid(self.masknet(torch.cat([I1tf, I2tf, feature], dim=1))).repeat(1, 3, 1, 1)

        # # situation when t_1=1, r=1
        # It_warp = ((1-t) * M * I1tf + t * (1 - M) * I2tf) / ((1-t) * M + t * (1-M)).clone()
        # # situation when t_0 + t_1 =1
        ratio_t_t1 = index_t
        It_warp = ((1-ratio_t_t1) * M * I1tf + ratio_t_t1 * (1 - M) * I2tf) / ((1-ratio_t_t1) * M + ratio_t_t1 * (1-M)).clone()

        return It_warp
