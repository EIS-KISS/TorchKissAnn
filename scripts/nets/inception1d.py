
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

"""
InceptionTime style multi scale convolutional network.

Each Inception module looks at the same part of the spectrum through several
convolution kernels at once (by default 5, 11 and 21 points) plus a 3 point
max pooling branch, and concatenates the results. A single kernel size forces
the network to encode a feature either as a narrow or as a wide pattern;
parallel kernels let it pick the scale that matches the feature, which matters
for spectra because a time constant shows up as an arc whose width in log
frequency varies by orders of magnitude between ECMs.

Reference:
 - H. Ismail Fawaz, B. Lucas, G. Forestier, C. Pelletier, D. F. Schmidt,
   J. Weber, G. I. Webb, L. Idoumghar, P.-A. Muller, F. Petitjean,
   "InceptionTime: Finding AlexNet for Time Series Classification", Data Mining
   and Knowledge Discovery 34, 1936-1962 (2020), doi:10.1007/s10618-020-00683-1,
   arXiv:1909.04939

Tuning note: this is the most learning-rate sensitive net of the set. On a 93
spectra regression dataset it needed 2e-4 to reach the same test loss the other
nets reach at the toolchain default of 5e-3, and diverged at 5e-3. Start
torchkissann_tune with --learning-rates 0.0002,0.001 for it.
"""

import torch

from .common import SamePadConv1d, spectra_length, spectra_to_channels


class InceptionModule(torch.nn.Module):
	"""
	Parallel convolutions of different width plus a pooled branch, concatenated,
	then batch norm and ReLU. Output width is n_filters * (len(kernel_sizes) + 1).

	The pooling branch is a 3 point max pooling with stride 1, so it keeps the
	length of the input and adds a max filtered view of it. torch.jit.script()
	does not allow a submodule that only exists in one branch of an if in the
	constructor, so the branch is always built.
	"""

	def __init__(self, in_channels: int, n_filters: int, kernel_sizes: list):
		super(InceptionModule, self).__init__()

		self.convs = torch.nn.ModuleList()
		for k in kernel_sizes:
			self.convs.append(SamePadConv1d(in_channels, n_filters, k, bias=False))

		self.pool_conv = SamePadConv1d(in_channels, n_filters, 1, bias=False)

		out_channels = n_filters * (len(kernel_sizes) + 1)
		self.bn = torch.nn.BatchNorm1d(out_channels)
		self.activation = torch.nn.ReLU(inplace=True)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		branches = [conv(x) for conv in self.convs]
		pooled = torch.nn.functional.max_pool1d(x, kernel_size=3, stride=1, padding=1)
		branches.append(self.pool_conv(pooled))
		x = torch.cat(branches, dim=1)
		x = self.bn(x)
		return self.activation(x)


class InceptionBlock(torch.nn.Module):
	"""Inception module with an optional residual connection."""

	def __init__(self, in_channels: int, out_channels: int, n_filters: int,
				 kernel_sizes: list, use_residual: bool):
		super(InceptionBlock, self).__init__()

		self.use_residual = use_residual
		self.module = InceptionModule(in_channels, n_filters, kernel_sizes)

		if use_residual and in_channels != out_channels:
			self.projection = torch.nn.Sequential(
				torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
				torch.nn.BatchNorm1d(out_channels))
		else:
			self.projection = torch.nn.Identity()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		out = self.module(x)
		if self.use_residual:
			out = out + self.projection(x)
		return out


class InceptionTime1D(torch.nn.Module):
	"""
	input_size is the length of the flat input vector (2N for a spectra with N
	frequency points), output_size the number of classes or regression targets.
	channels=2 splits the vector into an aligned Re(Z) and Im(Z) channel pair.
	"""

	def __init__(self, input_size: int, output_size: int, channels: int = 1,
				 n_filters: int = 32, n_modules: int = 4, kernel_sizes: list = [5, 11, 21]):
		super(InceptionTime1D, self).__init__()

		self.channels = channels
		length = spectra_length(input_size, channels)

		# Kernels wider than the input are pointless, drop duplicates that the
		# clamping produces.
		kernels = []
		for k in kernel_sizes:
			clamped = min(k, max(1, length))
			if clamped not in kernels:
				kernels.append(clamped)

		width = n_filters * (len(kernels) + 1)

		blocks = torch.nn.Sequential()
		for i in range(0, n_modules):
			in_channels = channels if i == 0 else width
			# Residual every third module, as in the paper. The first module
			# changes the number of channels, so it never gets a shortcut.
			use_residual = (i > 0) and ((i + 1) % 3 == 0)
			blocks.append(InceptionBlock(in_channels, width, n_filters, kernels, use_residual))

		self.blocks = blocks
		self.head = torch.nn.Sequential(
			torch.nn.AdaptiveAvgPool1d(1),
			torch.nn.Flatten(),
			torch.nn.Linear(width, output_size))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = spectra_to_channels(x, self.channels)
		x = self.blocks(x)
		return self.head(x)
