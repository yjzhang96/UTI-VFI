"""
Xiangyu Xu, Siyao Li, Wenxiu Sun, Qian Yin, and Ming-Hsuan Yang, "Quadratic Video Interpolation", NeurIPS 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from .UNet2 import UNet2 as UNet
import sys

class RefineS2(nn.Module):
    """Refine S1 from S-1, S0"""
    def __init__(self, path='./network-default.pytorch'):
        super(RefineS2, self).__init__()
       
        self.refinenet = UNet(6, 2)
        

    def forward(self, S0,S1,S2):

        # Input: S0-S2: (N, C, H, W)
        # Output: Refined S2
        S2, _ = self.refinenet(torch.cat((S0,S1,S2),dim=1))
        return S2