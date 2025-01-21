
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
from .simplenet import LinearBlock


def layer_width_function(layer: int, layers: int, input: int, output: int):
	layer_in = int((input * (layers - layer) / layers) + output * (layer) / layers)
	layer_out = int((input * (layers - layer - 1) / layers) + output * (layer + 1) / layers)
	return layer_in, layer_out


class MeanPool(torch.nn.Module):
	def __init__(self):
		super(MeanPool, self).__init__()

	def forward(self, x):
		x = x.mean(-1)
		return x


class ConvBlock(torch.nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
		super(ConvBlock, self).__init__()
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
		in_dim = net.shape[-1]
		out_dim = (in_dim + self.stride - 1) // self.stride
		p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
		pad_left = p // 2
		pad_right = p - pad_left
		net = torch.nn.functional.pad(net, (pad_left, pad_right), "constant", float(0))

		net = self.conv(net)

		return net


class Pool1dBlock(torch.nn.Module):
	def __init__(self, kernel_size):
		super(Pool1dBlock, self).__init__()
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


class ReshapeBlock(torch.nn.Module):
	def __init__(self):
		super(ReshapeBlock, self).__init__()

	def forward(self, x):
		if x.dim() == 2:
			x = x.reshape(x.size(0), 1, x.size(1))
		else:
			x = x.reshape(1, 1, x.size(0))
		return x


class UpsampleNet(torch.nn.Module):
	def __init__(self, input_size: int, output_size: int, downsample_steps: int, extra_steps: int):
		super(UpsampleNet, self).__init__()

		self.input_size = input_size
		self.layers = torch.nn.Sequential()

		for i in range(0, downsample_steps):
			layer_in, layer_out = layer_width_function(i, downsample_steps, input_size, output_size)
			self.layers.append(LinearBlock(layer_in, layer_out))
			if i == 0:
				for j in range(0, extra_steps):
					self.layers.append(LinearBlock(layer_out, layer_out))

		self.layers.append(ReshapeBlock())

		for i in range(0, downsample_steps):
			layer_in, layer_out = layer_width_function(i, downsample_steps, 1, output_size)
			self.layers.append(ConvBlock(layer_in, layer_out, 16, 1))
			self.layers.append(torch.nn.BatchNorm1d(layer_out))
			self.layers.append(torch.nn.ReLU())

			self.layers.append(ConvBlock(layer_out, layer_out, 16, 1))
			self.layers.append(torch.nn.BatchNorm1d(layer_out))
			self.layers.append(torch.nn.ReLU())

		self.layers.append(MeanPool())
		self.layers.append(LinearBlock(output_size, output_size))
		self.layers.append(LinearBlock(output_size, output_size))

	def forward(self, x: torch.Tensor):
		return self.layers.forward(x)
