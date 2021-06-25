import torch
import torch.nn as nn
from numpy.random import normal
from numpy.linalg import svd
from math import sqrt
import torch.nn.functional as F

def _get_orthogonal_init_weights(weights):
    fan_out = weights.size(0)
    fan_in = weights.size(1) * weights.size(2) * weights.size(3)

    u, _, v = svd(normal(0.0, 1.0, (fan_out, fan_in)), full_matrices=False)

    if u.shape == (fan_out, fan_in):
        return torch.Tensor(u.reshape(weights.size()))
    else:
        return torch.Tensor(v.reshape(weights.size()))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # m.weight.data.normal_(0.0, 0.02)
        nn.init.xavier_uniform_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def pixel_reshuffle(input, upscale_factor):
    """Rearranges elements in a tensor of shape ``[*, C, H, W]`` to a
	tensor of shape ``[C*r^2, H/r, W/r]``.

	See :class:`~torch.nn.PixelShuffle` for details.

	Args:
		input (Variable): Input
		upscale_factor (int): factor to increase spatial resolution by

	Examples:
		>>> input = autograd.Variable(torch.Tensor(1, 3, 12, 12))
		>>> output = pixel_reshuffle(input,2)
		>>> print(output.size())
		torch.Size([1, 12, 6, 6])
	"""
    batch_size, channels, in_height, in_width = input.size()

    # // division is to keep data type unchanged. In this way, the out_height is still int type
    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor
    input_view = input.contiguous().view(batch_size, channels, out_height, upscale_factor, out_width, upscale_factor)
    channels = channels * upscale_factor * upscale_factor

    shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return shuffle_out.view(batch_size, channels, out_height, out_width)


class RDB_block(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_block, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_block(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x

class Residual_Net(nn.Module):
    def __init__(self, in_channel, out_channel, n_RDB):
        super(Residual_Net, self).__init__()
        self.G0 = 96
        kSize = 3

        # number of RDB blocks, conv layers, out channels
        self.D = n_RDB
        self.C = 5
        self.G = 48

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(in_channel*4, self.G0, 5, padding=2, stride=1)
        self.SFENet2 = nn.Conv2d(self.G0, self.G0, kSize, padding=(kSize - 1) // 2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0=self.G0, growRate=self.G, nConvLayers=self.C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * self.G0, self.G0, 1, padding=0, stride=1),
            nn.Conv2d(self.G0, self.G0, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

        # Up-sampling net
        self.UPNet = nn.Sequential(*[
            nn.Conv2d(self.G0, 256, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.PixelShuffle(2),
            nn.Conv2d(64, out_channel, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

    def forward(self, input):
        B_shuffle = pixel_reshuffle(input, 2)
        f__1 = self.SFENet1(B_shuffle)
        x = self.SFENet2(f__1)
        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)
        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1

        # residual output
        Residual = self.UPNet(x) 
        return Residual

class Deblur_2step(nn.Module):
    def __init__(self, input_c, output_c=6, only_stage1=False):
        super(Deblur_2step,self).__init__()
        self.deblur_net = Residual_Net(input_c,12,n_RDB=20)
        self.refine_net = Residual_Net(9,output_c,n_RDB=10)
        self.only_stage1 = only_stage1

    def forward(self, B0,B1,B2,B3):
        input1 = torch.cat((B0,B1,B2,B3),1)
        # with torch.no_grad():
        if self.only_stage1:
            res1 = self.deblur_net(input1)
            deblur_out = torch.split(res1 + torch.cat((B1, B1, B2, B2), 1), 3, 1)
            return deblur_out
        else:
            res1 = self.deblur_net(input1).detach()
            deblur_out = torch.split(res1 + torch.cat((B1, B1, B2, B2), 1), 3, 1)
            deblur_B1 = torch.cat((B1, deblur_out[0],deblur_out[1]),1)
            deblur_B2 = torch.cat((B2, deblur_out[2],deblur_out[3]),1)
            res2_B1 = self.refine_net(deblur_B1)
            res2_B2 = self.refine_net(deblur_B2)
            refine_B1 = torch.split(res2_B1 + torch.cat((deblur_out[0], deblur_out[1]), 1), 3, 1)
            refine_B2 = torch.split(res2_B2 + torch.cat((deblur_out[2], deblur_out[3]), 1), 3, 1)
            refine_out = refine_B1 + refine_B2

            return deblur_out, refine_out
