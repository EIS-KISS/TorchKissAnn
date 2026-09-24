
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

from .common import SamePadConv1d, MeanPool, spectra_length, spectra_to_channels


class ResidualBlock(torch.nn.Module):

	def __init__(self, in_channels: int, out_channels: int, stride: int):
		super(ResidualBlock, self).__init__()

		self.conv1 = SamePadConv1d(in_channels, out_channels, 3, stride=stride, bias=False)
		self.bn1 = torch.nn.BatchNorm1d(out_channels)
		self.conv2 = SamePadConv1d(out_channels, out_channels, 3, bias=False)
		self.bn2 = torch.nn.BatchNorm1d(out_channels)

		if stride != 1 or in_channels != out_channels:
			self.shortcut = torch.nn.Sequential(
				torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
				torch.nn.BatchNorm1d(out_channels))
		else:
			self.shortcut = torch.nn.Identity()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		identity = self.shortcut(x)
		out = self.conv1(x)
		out = self.bn1(out)
		out = torch.nn.functional.relu(out, inplace=True)
		out = self.conv2(out)
		out = self.bn2(out)
		out = out + identity
		return torch.nn.functional.relu(out, inplace=True)


class ResNetV21D(torch.nn.Module):
	def __init__(self, input_size: int, output_size: int, channels: int = 1,
				 base_filters: int = 32, blocks_per_stage: int = 2, n_stages: int = 3,
				 dropout: float = 0.0):
		super(ResNetV21D, self).__init__()

		self.channels = channels
		length = spectra_length(input_size, channels)

		self.stem = torch.nn.Sequential(
			SamePadConv1d(channels, base_filters, 5, bias=False),
			torch.nn.BatchNorm1d(base_filters),
			torch.nn.ReLU(inplace=True))

		stages = torch.nn.Sequential()
		filters = base_filters
		for s in range(0, n_stages):
			out_filters = base_filters * (2 ** s)
			# Only halve the frequency axis while there is something to halve.
			stride = 2 if (s > 0 and length >= 4) else 1
			if stride == 2:
				length = (length + 1) // 2

			blocks = torch.nn.Sequential()
			for b in range(0, blocks_per_stage):
				in_filters = filters if b == 0 else out_filters
				blocks.append(ResidualBlock(in_filters, out_filters,
					stride if b == 0 else 1))
			filters = out_filters
			stages.append(blocks)

		self.stages = stages
		self.pool = MeanPool()
		self.dropout = torch.nn.Dropout(p=dropout) if dropout > 0 else torch.nn.Identity()
		self.head = torch.nn.Linear(filters, output_size)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = spectra_to_channels(x, self.channels)
		x = self.stem(x)
		x = self.stages(x)
		x = self.pool(x)
		x = self.dropout(x)
		return self.head(x)
