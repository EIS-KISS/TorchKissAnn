
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

# K. Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder
# for Statistical Machine Translation", EMNLP 2014, pp. 1724-1734,
# doi:10.3115/v1/D14-1179, arXiv:1406.1078 - the gated recurrent unit.

import torch

from .common import spectra_to_channels


class AttentionPool(torch.nn.Module):
	def __init__(self, hidden: int):
		super(AttentionPool, self).__init__()
		self.project = torch.nn.Linear(hidden, hidden)
		self.score = torch.nn.Linear(hidden, 1, bias=False)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		weights = torch.softmax(self.score(torch.tanh(self.project(x))), dim=1)
		return (x * weights).sum(dim=1)


class BiGRUNet(torch.nn.Module):
	def __init__(self, input_size: int, output_size: int, channels: int = 1,
				 hidden: int = 64, n_layers: int = 1, dropout: float = 0.2):
		super(BiGRUNet, self).__init__()

		self.channels = channels
		self.gru = torch.nn.GRU(input_size=channels, hidden_size=hidden,
			num_layers=n_layers, batch_first=True, bidirectional=True,
			dropout=dropout if n_layers > 1 else 0.0)
		self.attention = AttentionPool(hidden * 2)
		self.dropout = torch.nn.Dropout(p=dropout)
		self.head = torch.nn.Linear(hidden * 2, output_size)

	def forward(self: "BiGRUNet", x: torch.Tensor) -> torch.Tensor:
		x = spectra_to_channels(x, self.channels)
		# (batch, channels, length) -> (batch, length, channels), one step per
		# frequency point.
		x = x.transpose(1, 2).contiguous()
		states, _ = self.gru(x)
		pooled = self.attention(states)
		pooled = self.dropout(pooled)
		return self.head(pooled)
