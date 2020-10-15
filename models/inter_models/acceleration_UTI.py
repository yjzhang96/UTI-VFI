# This file implements acceleration/velocity calculation

import torch
import torch.nn as nn


class AcFusionLayer(nn.Module):
	"""docstring for AcFusionLayer"""
	def __init__(self, ):
		super(AcFusionLayer, self).__init__()
	
	def forward(self, flo10, flo12, flo23, r, index):
		"""
			-- input: four flows
			-- output: center shift
		"""
		# # situation when r=1
		# return 0.5 * ((t + t**2)*flo12 - (t - t**2)*flo10), 0.5 * (((1 - t) + (1 - t)**2)*flo21 - ((1 - t) - (1 - t)**2)*flo23)
		
		# # Uncertain Time Interval
		t = r/(r+1) * index
		v = r * (-flo10) + (flo23 - flo10)/2
		a = (r+1) * (flo23 + flo10)
		f_1_t = v * t + 0.5 * a * (t**2)
		return f_1_t