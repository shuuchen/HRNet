'''
This file is an implementation of the following paper:

	Deep High-Resolution Representation Learning for Visual Recognition
	https://arxiv.org/pdf/1908.07919.pdf
	2020

'''
import torch

from torch import nn


BN_MOMENTUM = 0.1

class Conv(nn.Module):
	def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, relued=True):
		super(Conv, self).__init__()
		padding = (kernel_size - 1) // 2
		self.conv_bn = nn.Sequential(
				nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
				nn.BatchNorm2d(out_ch, momentum=BN_MOMENTUM))
		self.relu = nn.ReLU()
		self.relued = relued

	def forward(self, x):
		x = self.conv_bn(x)
		if self.relued:
			x = self.relu(x)
		return x


class BasicBlock(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(BasicBlock, self).__init__()
		self.conv = nn.Sequential(
				Conv(in_ch, out_ch),
				Conv(in_ch, out_ch, relued=False))
		self.relu = nn.ReLU()
	def forward(self, x):
		identity = x
		x = self.conv(x)
		x += identity
		return self.relu(x)


class Bottleneck(nn.Module):

	expansion = 4

	def __init__(self, in_ch, out_ch, downsampling=None):
		super(Bottleneck, self).__init__()
		self.conv = nn.Sequential(
				Conv(in_ch, out_ch, kernel_size=1),
				Conv(out_ch, out_ch),
				Conv(out_ch, out_ch * self.expansion, kernel_size=1, relued=False))
		self.relu = nn.ReLU()
		self.downsampling = downsampling

	def forward(self, x):
		identity = x
		x = self.conv(x)
		if self.downsampling:
			identity = self.downsampling(identity)
		x += identity
		return self.relu(x)


class UpSampling(nn.Module):
	def __init__(self, ch, up_factor):
		super(UpSampling, self).__init__()
		self.up_sampling = nn.Sequential(
				nn.Upsample(scale_factor=up_factor, mode='bilinear', align_corners=False),
				Conv(ch, ch // up_factor, 1, relued=False))
	def forward(self, x):
		return self.up_sampling(x)


class DownSampling(nn.Module):
	def __init__(self, ch, num_samplings):
		super(DownSampling, self).__init__()
		convs = []
		for i in range(num_samplings):
			relued = True if i < num_samplings - 1 else False
			convs.append(Conv(ch, ch * 2, 3, 2, relued=relued))
			ch *= 2
		self.down_sampling = nn.Sequential(*convs)

	def forward(self, x):
		return self.down_sampling(x)


class HRBlock(nn.Module):
	def __init__(self, ch, index, last_stage, block, num_conv_block_per_list=4):
		super(HRBlock, self).__init__()
		self.index = index
		self.last_stage = last_stage
		self.num_conv_block_per_list = num_conv_block_per_list
		self.relu = nn.ReLU()

		self.parallel_conv_lists = nn.ModuleList()
		for i in range(index):
			ch_i = ch * 2**i
			conv_list = []
			for j in range(num_conv_block_per_list):
				conv_list.append(block(ch_i, ch_i))
			self.parallel_conv_lists.append(nn.Sequential(*conv_list))

		self.up_conv_lists = nn.ModuleList()
		for i in range(index - 1):
			conv_list = nn.ModuleList()
			for j in range(i + 1, index):
				up_factor = 2 ** (j-i)
				ch_j = ch * 2**j
				conv_list.append(UpSampling(ch_j, up_factor))
			self.up_conv_lists.append(conv_list)

		self.down_conv_lists = nn.ModuleList()
		for i in range(1, index if last_stage else index + 1):
			conv_list = nn.ModuleList()
			for j in range(i):
				ch_j = ch * 2**j
				conv_list.append(DownSampling(ch_j, i - j))
			self.down_conv_lists.append(conv_list)

	def forward(self, x_list):
		parallel_res_list = []
		for i in range(self.index):
			x = x_list[i]
			x = self.parallel_conv_lists[i](x)
			parallel_res_list.append(x)

		final_res_list = []
		for i in range(self.index if self.last_stage else self.index + 1):
			if i == self.index:
				x = 0
				for t, m in zip(parallel_res_list, self.down_conv_lists[-1]):
					x += m(t)
			else:
				x = parallel_res_list[i]
				if i != self.index - 1:
					res_list = parallel_res_list[i+1:]
					up_x = 0
					for t, m in zip(res_list, self.up_conv_lists[i]):
						up_x += m(t)
					x += up_x
				if i != 0:
					res_list = parallel_res_list[:i]
					down_x = 0
					for t, m in zip(res_list, self.down_conv_lists[i - 1]):
						down_x += m(t)
					x += down_x
			x = self.relu(x)
			final_res_list.append(x)
		return final_res_list


class HRNet(nn.Module):
	def __init__(self, in_ch, mid_ch, out_ch, num_stage=4):
		super(HRNet, self).__init__()
		self.init_conv = nn.Sequential(
					Conv(in_ch, 64, 1),
					Conv(64, 64, 1))
		self.head = nn.Sequential(
					Conv(mid_ch * (1 + 2 + 4 + 8), mid_ch * (1 + 2 + 4 + 8), 1),
					nn.Conv2d(mid_ch * (1 + 2 + 4 + 8), out_ch, 1))
		self.first_layer = self._make_layer(64, 64, Bottleneck, 4)
		self.first_transition = self._make_transition_layer(256, mid_ch, 1)
		self.num_stage = num_stage
		self.hr_blocks = nn.ModuleList()
		for i in range(1, num_stage):
			self.hr_blocks.append(HRBlock(mid_ch, i + 1, True if i == num_stage - 1 else False, BasicBlock))

		self.up_samplings = nn.ModuleList()
		for i in range(num_stage - 1):
			up_factor = 2 ** (i + 1)
			up = nn.Upsample(scale_factor=up_factor, mode='bilinear')
			self.up_samplings.append(up)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.normal_(m.weight, std=0.001)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def _make_layer(self, in_ch, ch, block, num):
		downsampling = None
		if in_ch != ch * block.expansion:
			downsampling = Conv(in_ch, ch * block.expansion, 1, relued=False)
		layers = []
		layers.append(block(in_ch, ch, downsampling))
		for i in range(1, num):
			layers.append(block(ch * block.expansion, ch))
		return nn.Sequential(*layers)
	
	def _make_transition_layer(self, in_ch, out_ch, stage):
		layers = nn.ModuleList()
		layers.append(Conv(in_ch, out_ch, 1))
		layers.append(Conv(in_ch, out_ch * 2, 3, 2))
		return layers

	def forward(self, x):
		x = self.init_conv(x)
		x = self.first_layer(x)
		x_list = [m(x) for m in self.first_transition]
		for i in range(self.num_stage - 1):
			x_list = self.hr_blocks[i](x_list)

		res_list = [x_list[0]]
		for t, m in zip(x_list[1:], self.up_samplings):
			res_list.append(m(t))
		x = torch.cat(res_list, dim=1)
		x = self.head(x)
		return x
