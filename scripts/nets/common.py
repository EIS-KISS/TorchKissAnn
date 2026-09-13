
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
Building blocks shared by the network definitions in this directory.

Everything here is written so that torch.jit.script() accepts it, because the
networks are consumed as TorchScript by ann::ScriptNet.
"""

import torch


def spectra_to_channels(x: torch.Tensor, channels: int) -> torch.Tensor:
	"""
	Turn the flat input vector produced by the dataset loaders into a
	(batch, channels, length) tensor.

	The loaders store a spectra as N real parts followed by N imaginary parts
	(see eisToTorch() in src/data/eistotorch.cpp), so splitting the vector in
	half gives two aligned channels: channel 0 is Re(Z), channel 1 is Im(Z).
	That lets a convolution see both components of the same frequency at once,
	which is what a local feature of a spectrum (the slope of the Nyquist arc,
	the height of a semicircle) actually is.

	Using paired real/imaginary channels instead of one long channel is the
	real valued form of feeding the complex impedance into the network, see:
	 - C. Trabelsi et al., "Deep Complex Networks", ICLR 2018, arXiv:1705.09792
	and for EIS networks that hand the components of Z to the first layer as
	separate channels rather than as one long vector:
	 - Q. Hua, M. Shen, "Deep Learning-Enhanced Parameter Extraction for
	   Equivalent Circuit Modeling in Electrochemical Impedance Spectroscopy",
	   NorCAS 2023, art. 10305482, doi:10.1109/NorCAS58970.2023.10305482 -
	   feeds real, imaginary, absolute and phase as four input channels to a 1D
	   CNN.

	channels=1 keeps the behaviour of the older nets (ConvNet, ResNet1D): the
	whole vector becomes a single channel, so Re and Im are only related through
	convolutions that span the middle of the vector.

	Note that channels=2 requires the input vector to hold nothing but the
	spectra. Datasets with extra scalar inputs append them behind the imaginary
	part, which would break the alignment - use channels=1 for those.
	"""
	if x.dim() == 1:
		x = x.reshape(1, -1)
	batch = x.size(0)
	if channels == 1:
		return x.reshape(batch, 1, x.size(1))
	elif channels == 2:
		length = x.size(1)
		if length % 2 != 0:
			raise ValueError("input size " + str(length) + " can not be split into a real and a imaginary channel")
		return x.reshape(batch, 2, length // 2)
	else:
		raise ValueError("only 1 or 2 input channels are supported, got " + str(channels))


def spectra_length(input_size: int, channels: int) -> int:
	"""Number of frequency points in the input, given the input vector length."""
	if channels == 2:
		if input_size % 2 != 0:
			raise ValueError("input size " + str(input_size) + " can not be split into a real and a imaginary channel")
		return input_size // 2
	return input_size


class SamePadConv1d(torch.nn.Module):
	"""
	Conv1d that pads the input so that the output length is ceil(length / stride),
	independent of the kernel size and dilation. The padding is computed at
	runtime from the tensor shape, so these networks are not tied to one input
	length (like the ones in resnet1d.py).
	"""

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
	"""Global average pooling over the frequency axis: (batch, C, L) -> (batch, C)."""

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return x.mean(-1)


class SqueezeExcitation1d(torch.nn.Module):
	"""
	Channel attention over the frequency axis.

	The squeeze is a global average pool over frequency, the excitation is a
	small bottleneck MLP producing one sigmoid weight per channel. For a
	spectrum this lets the network turn down bands that carry no information
	for the current task (e.g. the inductive tail) instead of having to learn
	to ignore them implicitly.

	Reference:
	 - J. Hu, L. Shen, G. Sun, "Squeeze-and-Excitation Networks", CVPR 2018,
	   pp. 7132-7141, arXiv:1709.01507
	"""

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
