import torch
from .simplenet import LinearBlock


def layer_width_function(layer: int, layers: int, input: int, output: int):
	layer_in = int((input * (layers - layer) / layers) + output * (layer) / layers)
	layer_out = int((input * (layers - layer - 1) / layers) + output * (layer + 1) / layers)
	return layer_in, layer_out


class MeanPool(torch.nn.Module):
	def __init__(self):
		super(MeanPool, self).__init__()

	def forward(self, x):
		x = x.mean(-1)
		return x


class ConvBlock(torch.nn.Module):
	def __init__(self, input_size: int, in_channels: int, out_channels: int, kernel_size: int, pad: bool = False):
		super(ConvBlock, self).__init__()
		self.input_size = input_size
		self.kernel_size = kernel_size
		self.pad = pad
		self.bn = torch.nn.BatchNorm1d(out_channels)
		self.do = torch.nn.Dropout(p=0.5)
		self.in_channels = in_channels
		self.out_channels = out_channels

		self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1)
		self.linear = torch.nn.Linear(self.outputSize(), self.outputSize())
		self.activation = torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)

	def outputSize(self) -> int:
		if not self.pad:
			return int(self.input_size - self.kernel_size + 1)
		else:
			return int(self.input_size)

	def forward(self, x):
		x = self.conv.forward(x)
		x = self.activation.forward(x)
		if self.pad:
			left_pad = int(self.kernel_size / 2)
			right_pad = int(self.kernel_size / 2) - 1
			x = torch.nn.functional.pad(x, (left_pad, right_pad), "constant", float(0))

		x = self.linear.forward(x)
		x = self.activation.forward(x)
		x = self.bn(x)
		return x


class ConvNet(torch.nn.Module):
	def __init__(self, input_size: int, output_size: int, downsample_steps: int, extra_steps: int):
		super(ConvNet, self).__init__()

		self.input_size = input_size
		self.layers = torch.nn.Sequential()

		filteres = 50

		self.layers.append(ConvBlock(input_size, 1, filteres, 16, True))
		self.layers.append(ConvBlock(self.layers[-1].outputSize(), filteres, filteres, 8, True))
		self.layers.append(ConvBlock(self.layers[-1].outputSize(), filteres, filteres, 8, True))
		self.layers.append(ConvBlock(self.layers[-1].outputSize(), filteres, filteres, 8, False))
		self.layers.append(ConvBlock(self.layers[-1].outputSize(), filteres, filteres, 8, False))
		self.layers.append(MeanPool())

		for i in range(0, downsample_steps):
			layer_in, layer_out = layer_width_function(i, downsample_steps, filteres, output_size)
			self.layers.append(LinearBlock(layer_in, layer_out))
			if i == 0:
				for j in range(0, extra_steps):
					self.layers.append(LinearBlock(layer_out, layer_out))

	def forward(self, x: torch.Tensor):
		if x.dim() == 2:
			x = x.reshape(x.size(0), 1, x.size(1))
		else:
			x = x.reshape(1, 1, x.size(0))
		return self.layers.forward(x)
