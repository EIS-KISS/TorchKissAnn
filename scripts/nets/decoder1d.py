
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


class ConvBlock(torch.nn.Module):
	def __init__(self, channels: int):
		super(ConvBlock, self).__init__()
		self.conv = torch.nn.Conv1d(channels, channels, kernel_size=5, padding=2)
		self.norm = torch.nn.BatchNorm1d(channels, eps=1e-3, momentum=0.1)
		self.act = torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.act(self.norm(self.conv(x)))


class DecoderNet(torch.nn.Module):

	def __init__(self, input_size: int, output_size: int, channels: int = 32, blocks: int = 4):
		super(DecoderNet, self).__init__()
		self.length = output_size // 2
		self.channels = channels
		self.proj = torch.nn.Linear(input_size, channels * self.length)
		stack: list = []
		for _ in range(blocks):
			stack.append(ConvBlock(channels))
		self.blocks = torch.nn.Sequential(*stack)
		self.head = torch.nn.Conv1d(channels, 2, kernel_size=5, padding=2)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = torch.nn.functional.leaky_relu(self.proj(x), 0.1)
		x = x.reshape(x.size(0), self.channels, self.length)
		x = self.blocks.forward(x)
		x = self.head(x)
		return x.reshape(x.size(0), 2 * self.length)
