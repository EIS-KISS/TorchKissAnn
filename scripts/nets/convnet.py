import torch
from .simplenet import LinearBlock


def layer_width_function(layer: int, layers: int, input: int, output: int):
	layer_in = int((input * (layers - layer) / layers) + output * (layer) / layers)
	layer_out = int((input * (layers - layer - 1) / layers) + output * (layer + 1) / layers)
	return layer_in, layer_out


class ConvBlock(torch.nn.Module):
	def __init__(self, input_size: int, in_channels: int, out_channels: int, kernel_size: int, devider: int):
		super(ConvBlock, self).__init__()
		self.input_size = input_size
		self.devider = devider
		self.kernel_size = kernel_size

		self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, bias=False)
		self.maxpool = torch.nn.MaxPool1d(devider)
		self.linear = torch.nn.Linear(self.outputSize(), self.outputSize())
		self.activation = torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)

	def outputSize(self) -> int:
		return int((self.input_size - self.kernel_size + 1) / self.devider)

	def forward(self, x):
		x = self.conv.forward(x)
		x = self.activation.forward(x)
		x = self.maxpool.forward(x)
		x = self.linear.forward(x)
		x = self.activation.forward(x)
		return x


class ConvNet(torch.nn.Module):
	def __init__(self, input_size: int, output_size: int, downsample_steps: int, extra_steps: int):
		super(ConvNet, self).__init__()

		conv1Kernel = int(input_size / 5)
		conv2Kernel = int(input_size / 10)

		self.input_size = input_size
		self.layers = torch.nn.Sequential()

		self.layers.append(ConvBlock(input_size, 1, 3, conv1Kernel, 2))
		self.layers.append(ConvBlock(self.layers[-1].outputSize(), 3, 6, conv1Kernel, 2))
		self.layers.append(torch.nn.Flatten())
		size = self.layers[-2].outputSize() * 6

		for i in range(0, downsample_steps):
			layer_in, layer_out = layer_width_function(i, downsample_steps, size, output_size)
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
