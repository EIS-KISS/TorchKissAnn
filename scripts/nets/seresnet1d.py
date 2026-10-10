
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
Compact residual network with channel attention (SE-ResNet in 1D).

A small ResNet: a stem convolution, then three stages that double the number of
filters while halving the number of frequency points, each stage made of basic
residual blocks. Every block ends with a squeeze-and-excitation unit, which
reweights the filters by how much of the spectrum they fired on. Two things
make this attractive for EIS: the residual connections keep a network this deep
trainable on a few hundred spectra, and the SE weights say which filters the
network actually uses, which is cheap to inspect after training.

This is the compact counterpart to resnet1d.py, which uses 100 base filters and
about 10M parameters regardless of the problem size - far more than the small
generated datasets here can justify.

References:
 - K. He, X. Zhang, S. Ren, J. Sun, "Deep Residual Learning for Image
   Recognition", CVPR 2016, pp. 770-778, doi:10.1109/CVPR.2016.90,
   arXiv:1512.03385 - residual blocks and the stage/doubling layout.
 - J. Hu, L. Shen, G. Sun, "Squeeze-and-Excitation Networks", CVPR 2018,
   pp. 7132-7141, doi:10.1109/CVPR.2018.00745, arXiv:1709.01507 - the SE unit.
 - S. Zagoruyko, N. Komodakis, "Wide Residual Networks", arXiv:1605.07146 -
   width, not depth, is what a small dataset benefits from; the base filter
   count here is a knob for exactly that.

For deep learning on EIS spectra in this benchmark of equivalent circuit model
classification (which includes a 1D convolutional network among the learned
baselines):
 - J. Schaeffer et al., "Machine Learning Benchmarks for the Classification of
   Equivalent Circuit Models from Electrochemical Impedance Spectra", Journal
   of The Electrochemical Society 170, 060512 (2023), doi:10.1149/1945-7111/acd8fb,
   arXiv:2302.03362
"""

import torch

from .common import SamePadConv1d, SqueezeExcitation1d, MeanPool, spectra_length, spectra_to_channels, extra_inputs_slice


class ResidualBlock(torch.nn.Module):
	"""conv-bn-relu-conv-bn-se with a shortcut, optionally striding by 2."""

	def __init__(self, in_channels: int, out_channels: int, stride: int, reduction: int):
		super(ResidualBlock, self).__init__()

		self.conv1 = SamePadConv1d(in_channels, out_channels, 3, stride=stride, bias=False)
		self.bn1 = torch.nn.BatchNorm1d(out_channels)
		self.conv2 = SamePadConv1d(out_channels, out_channels, 3, bias=False)
		self.bn2 = torch.nn.BatchNorm1d(out_channels)
		self.se = SqueezeExcitation1d(out_channels, reduction)

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
		out = self.se(out)
		out = out + identity
		return torch.nn.functional.relu(out, inplace=True)


class SEResNet1D(torch.nn.Module):
	"""
	input_size is the length of the flat input vector, output_size the number of
	classes or regression targets. channels=2 splits the spectra into an aligned
	Re(Z) and Im(Z) channel pair.

	extra_count is the number of trailing scalar inputs (temperature, ocv, ...)
	the loader appended behind the spectra. When it is non zero the network
	becomes two-path: the spectra part goes through the convolution trunk as
	before, and the extra scalars go through a small MLP (extra_width wide); the
	pooled conv feature and the MLP feature are concatenated before the head.
	This keeps the scalars out of the frequency axis, so they are neither
	convolved against a neighbouring frequency nor diluted by the global mean
	pool the way they would be if they were left inside the conv input. With
	extra_count=0 the network is exactly the single-path SEResNet1D.
	"""

	def __init__(self, input_size: int, output_size: int, channels: int = 1,
				 base_filters: int = 32, blocks_per_stage: int = 2, n_stages: int = 3,
				 reduction: int = 4, dropout: float = 0.0,
				 extra_count: int = 0, extra_width: int = 16):
		super(SEResNet1D, self).__init__()

		self.channels: torch.jit.Final[int] = channels
		self.extra_count: torch.jit.Final[int] = extra_count
		length = spectra_length(input_size, channels, extra_count)

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
					stride if b == 0 else 1, reduction))
			filters = out_filters
			stages.append(blocks)

		self.stages = stages
		self.pool = MeanPool()
		self.dropout = torch.nn.Dropout(p=dropout) if dropout > 0 else torch.nn.Identity()

		head_in = filters
		if extra_count > 0:
			self.extra = torch.nn.Sequential(
				torch.nn.Linear(extra_count, extra_width),
				torch.nn.ReLU(inplace=True),
				torch.nn.Linear(extra_width, extra_width))
			head_in = filters + extra_width
		else:
			# TorchScript resolves self.extra even in the pruned branch, so it must exist;
			# Identity keeps the single-path net byte-for-byte the old one (no extra params).
			self.extra = torch.nn.Identity()
		self.head = torch.nn.Linear(head_in, output_size)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		flat = x
		x = spectra_to_channels(flat, self.channels, self.extra_count)
		x = self.stem(x)
		x = self.stages(x)
		x = self.pool(x)
		x = self.dropout(x)
		if self.extra_count > 0:
			extra = self.extra(extra_inputs_slice(flat, self.extra_count))
			x = torch.cat([x, extra], dim=1)
		return self.head(x)
