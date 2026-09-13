
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
Bidirectional GRU with attention pooling.

The spectrum is walked once from high to low frequency and once back, so every
frequency point is described by what came before and after it in log frequency.
The hidden states of all points are then combined by additive attention instead
of a mean: the network learns which frequency ranges are worth reading for the
task at hand, which is a different inductive bias from the convolutional nets
in this directory - a convolution fixes the neighbourhood, the recurrent state
carries an unbounded one at the price of a sequential computation.

References:
 - K. Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder
   for Statistical Machine Translation", EMNLP 2014, pp. 1724-1734,
   doi:10.3115/v1/D14-1179, arXiv:1406.1078 - the gated recurrent unit.
 - M. Schuster, K. K. Paliwal, "Bidirectional Recurrent Neural Networks",
   IEEE Transactions on Signal Processing 45(11), 1997, pp. 2673-2681,
   doi:10.1109/78.650093 - running the recurrence in both directions.
 - D. Bahdanau, K. Cho, Y. Bengio, "Neural Machine Translation by Jointly
   Learning to Align and Translate", ICLR 2015, arXiv:1409.0473 - additive
   (softmax weighted) attention, used here as a pooling over frequency.
 - S. Bai, J. Z. Kolter, V. Koltun, "An Empirical Evaluation of Generic
   Convolutional and Recurrent Networks for Sequence Modeling", arXiv:1803.01271 -
   the counter position: convolutions often match recurrent nets, so this model
   is worth including mainly as the comparison to the TCN and Inception nets.

For EIS specifically, learned models that read a whole spectrum and decide
between equivalent circuit models or predict their parameters are established,
e.g.:
 - S. Zhu, X. Sun, X. Gao, J. Wang, N. Zhao, J. Sha, "Equivalent circuit model
   recognition of electrochemical impedance spectroscopy via machine learning",
   Journal of Electroanalytical Chemistry 855, 113627 (2019),
   doi:10.1016/j.jelechem.2019.113627
"""

import torch

from .common import spectra_to_channels


class AttentionPool(torch.nn.Module):
	"""
	Additive attention over the sequence axis: (batch, L, H) -> (batch, H).

	Scores each hidden state with a one dimensional softmax weight and returns
	the weighted sum, so the output length no longer depends on the number of
	frequency points.
	"""

	def __init__(self, hidden: int):
		super(AttentionPool, self).__init__()
		self.project = torch.nn.Linear(hidden, hidden)
		self.score = torch.nn.Linear(hidden, 1, bias=False)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		weights = torch.softmax(self.score(torch.tanh(self.project(x))), dim=1)
		return (x * weights).sum(dim=1)


class BiGRUNet(torch.nn.Module):
	"""
	input_size is the length of the flat input vector (2N for a spectra with N
	frequency points), output_size the number of classes or regression targets.
	channels=2 splits the vector into an aligned Re(Z) and Im(Z) channel pair,
	which is the natural input for a recurrent net: one step per frequency point
	carrying both components.
	"""

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
