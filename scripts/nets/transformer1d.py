
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
Transformer encoder over frequency "tokens".

Each frequency point (or a small patch of consecutive points) is projected to a
d_model dimensional vector, a learned positional embedding is added because
self attention is otherwise blind to the order of the points, and the sequence
goes through a pre-norm transformer encoder. The representation is pooled over
the frequency axis and read out by one linear layer.

Attention connects any two frequency points in a single step, so relations like
"the high frequency intercept is large while the mid frequency arc is flat"
cost nothing, unlike in a convolution stack where they need many layers. The
price is quadratic cost in the number of points and a strong tendency to
overfit small datasets, so this is a candidate for the larger generated
datasets, not the small ones.

References:
 - A. Vaswani et al., "Attention is All you Need", NeurIPS 30, 2017,
   arXiv:1706.03762 - the transformer encoder.
"""

import torch

from .common import spectra_length, spectra_to_channels


class Transformer1D(torch.nn.Module):
	"""
	input_size is the length of the flat input vector (2N for a spectra with N
	frequency points), output_size the number of classes or regression targets.
	channels=2 splits the vector into an aligned Re(Z) and Im(Z) channel pair,
	so every token carries both components of its frequency point.
	"""

	def __init__(self, input_size: int, output_size: int, channels: int = 1,
				 d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
				 dim_feedforward: int = 128, dropout: float = 0.1, patch_size: int = 1):
		super(Transformer1D, self).__init__()

		if d_model % n_heads != 0:
			raise ValueError("d_model " + str(d_model) + " must be divisible by n_heads " + str(n_heads))

		self.channels = channels
		length = spectra_length(input_size, channels)
		patch = min(patch_size, max(1, length))
		n_tokens = (length - patch) // patch + 1

		# Patching: a strided convolution turns each patch of `patch` frequency
		# points into one token. With patch_size=1 this is a per point linear
		# projection, which is the cheapest useful setting for the short
		# sequences EIS datasets have.
		self.embed = torch.nn.Conv1d(channels, d_model, kernel_size=patch, stride=patch)
		self.embed_norm = torch.nn.LayerNorm(d_model)

		# Learned positional embedding: attention permutes its input, so without
		# this the network could not tell a high frequency arc from a low
		# frequency one.
		self.position = torch.nn.Parameter(torch.zeros(1, n_tokens, d_model))
		torch.nn.init.normal_(self.position, mean=0.0, std=0.02)

		layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
			dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True, norm_first=True)
		self.encoder = torch.nn.TransformerEncoder(layer, num_layers=n_layers)

		self.norm = torch.nn.LayerNorm(d_model)
		self.head = torch.nn.Linear(d_model, output_size)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = spectra_to_channels(x, self.channels)
		x = self.embed(x).transpose(1, 2).contiguous()
		x = self.embed_norm(x) + self.position
		x = self.encoder(x)
		# Pool over frequency instead of reading a classification token: with a
		# few hundred training spectra the pooled mean is the more stable readout.
		x = x.mean(1)
		x = self.norm(x)
		return self.head(x)
