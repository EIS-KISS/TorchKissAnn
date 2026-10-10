
# TorchKissAnn - A collection of tools to train various types of Machine learning
# algorithms on various types of EIS data
# Copyright (C) 2026 Carl Klemm <carl@uvos.xyz>
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


def spectra_to_channels(x: torch.Tensor, channels: int, extra_count: int = 0) -> torch.Tensor:
	if x.dim() == 1:
		x = x.reshape(1, -1)
	batch = x.size(0)
	if extra_count > 0:
		x = x[:, :x.size(1) - extra_count]
	if channels == 1:
		return x.reshape(batch, 1, x.size(1))
	elif channels == 2:
		length = x.size(1)
		if length % 2 != 0:
			raise ValueError("input size " + str(length) + " can not be split into a real and a imaginary channel")
		return x.reshape(batch, 2, length // 2)
	else:
		raise ValueError("only 1 or 2 input channels are supported, got " + str(channels))


def spectra_length(input_size: int, channels: int, extra_count: int = 0) -> int:
	spectral = input_size - extra_count
	if channels == 2:
		if spectral % 2 != 0:
			raise ValueError("input size " + str(input_size) + " can not be split into a real and a imaginary channel")
		return spectral // 2
	return spectral


def extra_inputs_slice(x: torch.Tensor, extra_count: int) -> torch.Tensor:
	"""The trailing extra_count scalars of the flat input vector, as (batch, extra_count)."""
	if x.dim() == 1:
		x = x.reshape(1, -1)
	return x[:, x.size(1) - extra_count:]


class SamePadConv1d(torch.nn.Module):
	def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
				 dilation: int = 1, stride: int = 1, groups: int = 1, bias: bool = True):
		super(SamePadConv1d, self).__init__()
		self.kernel_size = kernel_size
		self.dilation = dilation
		self.stride = stride
		self.conv = torch.nn.Conv1d(
			in_channels=in_channels,
			out_channels=out_channels,
			kernel_size=kernel_size,
			dilation=dilation,
			stride=stride,
			groups=groups,
			bias=bias)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		in_dim = x.shape[-1]
		out_dim = (in_dim + self.stride - 1) // self.stride
		p = max(0, (out_dim - 1) * self.stride + (self.kernel_size - 1) * self.dilation + 1 - in_dim)
		pad_left = p // 2
		pad_right = p - pad_left
		x = torch.nn.functional.pad(x, (pad_left, pad_right), "constant", float(0))
		return self.conv(x)


class MeanPool(torch.nn.Module):
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return x.mean(-1)


class SqueezeExcitation1d(torch.nn.Module):
	def __init__(self, channels: int, reduction: int = 4):
		super(SqueezeExcitation1d, self).__init__()
		hidden = max(1, channels // reduction)
		self.fc1 = torch.nn.Linear(channels, hidden)
		self.fc2 = torch.nn.Linear(hidden, channels)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		squeezed = x.mean(-1)
		excited = torch.nn.functional.relu(self.fc1(squeezed))
		excited = torch.sigmoid(self.fc2(excited))
		return x * excited.unsqueeze(-1)
