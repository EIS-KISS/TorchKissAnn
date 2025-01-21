
# TorchKissAnn - A collection of tools to train various types of Machine learning
# algorithms on various types of EIS data
# Copyright (C) 2025 Carl Klemm <carl@uvos.xyz>
#
# This file is part of TorchKissAnn.
#
# TorchKissAnn is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# TorchKissAnn is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with TorchKissAnn.  If not, see <http://www.gnu.org/licenses/>.

import torch


class Conv1dSamePad(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
		super(Conv1dSamePad, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.stride = stride
		self.groups = groups
		self.conv = torch.nn.Conv1d(
			in_channels=self.in_channels,
			out_channels=self.out_channels,
			kernel_size=self.kernel_size,
			stride=self.stride,
			groups=self.groups)

	def forward(self, x):
		net = x
		# compute pad shape
		in_dim = net.shape[-1]
		out_dim = (in_dim + self.stride - 1) // self.stride
		p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
		pad_left = p // 2
		pad_right = p - pad_left
		net = torch.nn.functional.pad(net, (pad_left, pad_right), "constant", float(0))

		net = self.conv(net)

		return net


class MaxpoolSamePad(nn.Module):
	def __init__(self, kernel_size):
		super(MaxpoolSamePad, self).__init__()
		self.kernel_size = kernel_size
		self.stride = 1
		self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

	def forward(self, x):
		net = x
		in_dim = net.shape[-1]
		out_dim = (in_dim + self.stride - 1) // self.stride
		p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
		pad_left = p // 2
		pad_right = p - pad_left
		net = torch.nn.functional.pad(net, (pad_left, pad_right), "constant", float(0))

		net = self.max_pool(net)

		return net


class ResnetBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, groups, downsample, use_bn, use_do, is_first_block=False):
		super(ResnetBlock, self).__init__()

		self.in_channels = in_channels
		self.kernel_size = kernel_size
		self.out_channels = out_channels
		self.stride = stride
		self.groups = groups
		self.downsample = downsample
		if self.downsample:
			self.stride = stride
		else:
			self.stride = 1
		self.is_first_block = is_first_block
		self.use_bn = use_bn
		self.use_do = use_do

		# the first conv
		self.bn1 = torch.nn.BatchNorm1d(in_channels)
		self.relu1 = torch.nn.ReLU()
		self.do1 = torch.nn.Dropout(p=0.5)
		self.conv1 = Conv1dSamePad(
			in_channels=in_channels,
			out_channels=out_channels,
			kernel_size=kernel_size,
			stride=self.stride,
			groups=self.groups)

		# the second conv
		self.bn2 = torch.nn.BatchNorm1d(out_channels)
		self.relu2 = torch.nn.ReLU()
		self.do2 = torch.nn.Dropout(p=0.5)
		self.conv2 = Conv1dSamePad(
			in_channels=out_channels,
			out_channels=out_channels,
			kernel_size=kernel_size,
			stride=1,
			groups=self.groups)

		self.max_pool = MaxpoolSamePad(kernel_size=self.stride)

	def forward(self, x):

		identity = x

		# the first conv
		out = x
		if not self.is_first_block:
			if self.use_bn:
				out = self.bn1(out)
			out = self.relu1(out)
			if self.use_do:
				out = self.do1(out)
		out = self.conv1(out)

		# the second conv
		if self.use_bn:
			out = self.bn2(out)
		out = self.relu2(out)
		if self.use_do:
			out = self.do2(out)
		out = self.conv2(out)

		# if downsample, also downsample identity
		if self.downsample:
			identity = self.max_pool(identity)

		# if expand channel, also pad zeros to identity
		if self.out_channels != self.in_channels:
			identity = identity.transpose(-1, -2)
			ch1 = (self.out_channels - self.in_channels) // 2
			ch2 = self.out_channels - self.in_channels - ch1
			identity = torch.nn.functional.pad(identity, (ch1, ch2), "constant", float(0))
			identity = identity.transpose(-1, -2)

		# shortcut
		out += identity

		return out


class ResNet1D(nn.Module):
	def __init__(self, in_channels, base_filters, kernel_size, stride, groups, n_classes,
					downsample_gap=2, increasefilter_gap=4, use_bn=True, use_do=True, verbose=False):
		super(ResNet1D, self).__init__()

		self.verbose = verbose
		self.n_block = 8
		self.kernel_size = kernel_size
		self.stride = stride
		self.groups = groups
		self.use_bn = use_bn
		self.use_do = use_do

		self.downsample_gap = downsample_gap
		self.increasefilter_gap = increasefilter_gap

		# first block
		self.first_block_conv = Conv1dSamePad(in_channels=in_channels, out_channels=base_filters, kernel_size=self.kernel_size, stride=1)
		self.first_block_bn = torch.nn.BatchNorm1d(base_filters)
		self.first_block_relu = torch.nn.ReLU()
		out_channels = base_filters

		# residual blocks
		self.basicblock_list = torch.nn.ModuleList()
		for i_block in range(self.n_block):
			# is_first_block
			if i_block == 0:
				is_first_block = True
			else:
				is_first_block = False
			# downsample at every self.downsample_gap blocks
			if i_block % self.downsample_gap == 1:
				downsample = True
			else:
				downsample = False
			# in_channels and out_channels
			if is_first_block:
				in_channels = base_filters
				out_channels = in_channels
			else:
				# increase filters at every self.increasefilter_gap blocks
				in_channels = int(base_filters * 2 ** ((i_block - 1) // self.increasefilter_gap))
				if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
					out_channels = in_channels * 2
				else:
					out_channels = in_channels

			tmp_block = ResnetBlock(
				in_channels=in_channels,
				out_channels=out_channels,
				kernel_size=self.kernel_size,
				stride=self.stride,
				groups=self.groups,
				downsample=downsample,
				use_bn=self.use_bn,
				use_do=self.use_do,
				is_first_block=is_first_block)
			self.basicblock_list.append(tmp_block)

		# final prediction
		self.final_bn = torch.nn.BatchNorm1d(out_channels)
		self.final_relu = torch.nn.ReLU(inplace=True)
		# self.do =torch.nn.Dropout(p=0.5)
		self.dense = torch.nn.Linear(out_channels, n_classes)
		# self.softmax =torch.nn.Softmax(dim=1)
		for i_block in range(self.n_block):
			net = self.basicblock_list[i_block]

	def forward(self, x):
		if x.dim() == 2:
			x = x.reshape(x.size(0), 1, x.size(1))
		else:
			x = x.reshape(1, 1, x.size(0))
		out = x

		# first conv
		if self.verbose:
			print('input shape', out.shape)
		out = self.first_block_conv(out)
		if self.verbose:
			print('after first conv', out.shape)
		if self.use_bn:
			out = self.first_block_bn(out)
		out = self.first_block_relu(out)

		# residual blocks, every block has two conv
		# fixed size due to a defficany in torch
		assert len(self.basicblock_list) == 8
		net = self.basicblock_list[0]
		out = net(out)
		net = self.basicblock_list[1]
		out = net(out)
		net = self.basicblock_list[2]
		out = net(out)
		net = self.basicblock_list[3]
		out = net(out)
		net = self.basicblock_list[4]
		out = net(out)
		net = self.basicblock_list[5]
		out = net(out)
		net = self.basicblock_list[6]
		out = net(out)
		net = self.basicblock_list[7]
		out = net(out)

		# final prediction
		if self.use_bn:
			out = self.final_bn(out)
		out = self.final_relu(out)
		out = out.mean(-1)
		if self.verbose:
			print('final pooling', out.shape)
		# out = self.do(out)
		out = self.dense(out)
		if self.verbose:
			print('dense', out.shape)
		# out = self.softmax(out)
		if self.verbose:
			print('softmax', out.shape)
		return out
