import torch


def layer_width_function(layer: int, layers: int, input: int, output: int):
	layer_in = int((input * (layers - layer) / layers) + output * (layer) / layers)
	layer_out = int((input * (layers - layer - 1) / layers) + output * (layer + 1) / layers)
	return layer_in, layer_out


class LinearBlock(torch.nn.Module):
	def __init__(self, input_size: int, output_size: int):
		super(LinearBlock, self).__init__()
		self.linear = torch.nn.Linear(input_size, output_size)
		self.activation = torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)
		self.batchnorm = torch.nn.BatchNorm1d(output_size, eps=1e-3, momentum=0.1, affine=True)

	def forward(self, x):
		x = self.linear.forward(x)
		x = self.activation.forward(x)
		x = self.batchnorm.forward(x)
		return x


class SimpleNet(torch.nn.Module):
	def __init__(self, input_size: int, output_size: int, downsample_steps: int, extra_steps: int):
		super(SimpleNet, self).__init__()

		self.input_size = input_size
		self.layers = torch.nn.Sequential()

		layer_in, layer_out = layer_width_function(0, downsample_steps, input_size, input_size)
		self.layers.append(LinearBlock(layer_in, layer_out))

		for i in range(0, downsample_steps):
			layer_in, layer_out = layer_width_function(i, downsample_steps, input_size, output_size)
			self.layers.append(LinearBlock(layer_in, layer_out))
			if i == 0:
				for j in range(0, extra_steps):
					self.layers.append(LinearBlock(layer_out, layer_out))

	def forward(self, x: torch.Tensor):
		return self.layers.forward(x)
