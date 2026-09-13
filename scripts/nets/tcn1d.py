
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
Temporal convolutional network (TCN): a stack of dilated residual blocks.

Every block uses the same small kernel (3 points) but an exponentially growing
dilation (1, 2, 4, 8, ...), so the receptive field doubles per block while the
parameter count stays flat. With 6 blocks a single output unit sees the whole
spectrum, which is what a Warburg tail or the offset between two time constants
needs, without the hundreds of thousands of parameters of a wide dense tail.

References:
 - C. Lea, M. D. Flynn, R. Vidal, A. Reiter, G. D. Hager, "Temporal
   Convolutional Networks for Action Segmentation and Detection", CVPR 2017,
   pp. 156-165, doi:10.1109/CVPR.2017.200, arXiv:1611.05285 - introduced the
   dilated causal convolution stack.
 - S. Bai, J. Z. Kolter, V. Koltun, "An Empirical Evaluation of Generic
   Convolutional and Recurrent Networks for Sequence Modeling", arXiv:1803.01271 -
   the simplified residual TCN used here, and the evidence that this class of
   convolutional network matches or beats LSTMs/GRUs on sequence tasks.
 - K. He, X. Zhang, S. Ren, J. Sun, "Deep Residual Learning for Image
   Recognition", CVPR 2016, pp. 770-778, arXiv:1512.03385 - residual blocks.

Dilation is capped at half the input length: a dilation larger than the input
spreads the kernel over the zero padding only and stops adding context.

Unlike the papers above the convolutions here are not causal. A spectrum is not
a time series that must be predicted from its past - the full Re/Im vector is
available at inference time - so each point may look at both neighbours, which
halves the depth needed to cover the input and keeps the phase of a feature
from drifting to one side.
"""

import torch

from .common import MeanPool, SamePadConv1d, spectra_length, spectra_to_channels


class DilatedResidualBlock(torch.nn.Module):
	"""Two dilated convolutions with a projection shortcut, BN + ReLU + dropout."""

	def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
				 dilation: int, dropout: float):
		super(DilatedResidualBlock, self).__init__()

		self.conv1 = SamePadConv1d(in_channels, out_channels, kernel_size, dilation=dilation, bias=False)
		self.bn1 = torch.nn.BatchNorm1d(out_channels)
		self.conv2 = SamePadConv1d(out_channels, out_channels, kernel_size, dilation=dilation, bias=False)
		self.bn2 = torch.nn.BatchNorm1d(out_channels)
		self.dropout = torch.nn.Dropout(p=dropout)

		# Normalises the result of the addition. Without it every block adds an
		# unbounded term to the previous activations and the stack drifts to
		# large outputs: with 6 blocks the logits start out around +-25, which
		# is an initial cross entropy of 40 or more and a needlessly unstable
		# start. The TCN papers keep the magnitude in check with weight
		# normalisation instead; batch norm after the sum does the same job
		# without an extra parameterisation.
		self.bn3 = torch.nn.BatchNorm1d(out_channels)

		if in_channels != out_channels:
			self.shortcut = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
		else:
			self.shortcut = torch.nn.Identity()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		identity = self.shortcut(x)
		out = self.conv1(x)
		out = self.bn1(out)
		out = torch.nn.functional.relu(out, inplace=True)
		out = self.dropout(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.bn3(out + identity)
		return torch.nn.functional.relu(out, inplace=True)


class TCN1D(torch.nn.Module):
	"""
	input_size is the length of the flat input vector (2N for a spectra with N
	frequency points), output_size the number of classes or regression targets.
	channels=2 splits the vector into an aligned Re(Z) and Im(Z) channel pair.
	"""

	def __init__(self, input_size: int, output_size: int, channels: int = 1,
				 hidden: int = 64, n_blocks: int = 6, kernel_size: int = 3,
				 dilation_base: int = 2, dropout: float = 0.2):
		super(TCN1D, self).__init__()

		self.channels = channels
		length = spectra_length(input_size, channels)

		blocks = torch.nn.Sequential()
		dilation = 1
		for i in range(0, n_blocks):
			in_channels = channels if i == 0 else hidden
			# Do not let the kernel reach further than the input exists.
			blocks.append(DilatedResidualBlock(in_channels, hidden, kernel_size,
				min(dilation, max(1, length // 2)), dropout))
			dilation = dilation * dilation_base

		self.blocks = blocks
		self.pool = MeanPool()
		self.head = torch.nn.Linear(hidden, output_size)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = spectra_to_channels(x, self.channels)
		x = self.blocks(x)
		x = self.pool(x)
		return self.head(x)
