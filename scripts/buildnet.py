
# TorchKissAnn - A collection of tools to train various types of Machine learning
# algorithms on various types of EIS data
# Copyright (C) 2025 Carl Klemm <carl@uvos.xyz>
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

import torch
from nets.resnet1d import ResNet1D
from nets.simplenet import SimpleNet
from nets.convnet import ConvNet
from nets.upsamplenet import UpsampleNet
import json
import argparse


def save(model, input_size, output_size, name: str):
	print(f"\n{name}:")
	print(model.code)
	print(f"Parameter Count: {sum(p.numel() for p in model.parameters())}")
	input_str = str(input_size) if input_size is not None else "ANY"

	outputLabels: list[str] = list()
	for i in range(0, output_size):
		outputLabels = outputLabels + ["class_" + str(i)]

	meta_json = json.dumps({
		'inputSize': input_size,
		'outputSize': output_size,
		'name': name,
		'outputLabels': outputLabels,
		'purpose': 'Unkown',
		'inputLabel': 'EIS'})

	print(meta_json)

	metadata = {'meta.json': meta_json}
	torch.jit.save(model, f"{name}{input_str}-{output_size}.pt", _extra_files=metadata)

	print(f"Saved model as ./{name}{input_str}-{output_size}.pt")


def build_resnet(output_size: int):
	module = ResNet1D(in_channels=1,
		base_filters=100,
		kernel_size=16,
		stride=2,
		groups=1,
		n_classes=output_size,
		downsample_gap=output_size,
		increasefilter_gap=12,
		verbose=False)

	data = torch.randn((16, 100))
	data = module.forward(data)
	print(f"ResNet1D output size: {data.size()}")

	script = torch.jit.script(module)
	save(script, None, output_size, "resnet")


def build_simplenet(input_size: int, output_size: int):
	module = SimpleNet(input_size, output_size, 4, 3)

	data = torch.randn((16, input_size))
	data = module.forward(data)
	print(f"SimpleNet output size: {data.size()}")

	script = torch.jit.script(module)

	save(script, input_size, output_size, "simplenet")


def build_convnet(input_size: int, output_size: int):
	module = ConvNet(input_size, output_size, 4, 3)

	data = torch.randn((16, input_size))
	data = module.forward(data)
	print(f"ConvNet output size: {data.size()}")

	script = torch.jit.script(module)

	save(script, input_size, output_size, "convnet")


def build_upsamplenet(input_size: int, output_size: int):
	module = UpsampleNet(input_size, output_size, 4, 3)

	data = torch.randn((16, input_size))
	data = module.forward(data)
	print(f"UpsampleNet output size: {data.size()}")

	script = torch.jit.script(module)

	save(script, input_size, output_size, "upsamplenet")


if __name__ == "__main__":
	valid_network_types = "simple, conv, resnet, upsamplenet"
	parser = argparse.ArgumentParser("TorchScript network generation script")
	parser.add_argument('--type', '-t', required=True, help=f"Type of network to create: {valid_network_types}")
	parser.add_argument('--output_size', '-o', type=int, required=True, help="Number of output classes")
	parser.add_argument('--input_size', '-i', type=int, default=100, help="Number of input classes")
	args = parser.parse_args()

	if args.type == "simple":
		build_simplenet(args.input_size, args.output_size)
	elif args.type == "conv":
		build_convnet(args.input_size, args.output_size)
	elif args.type == "resnet":
		build_resnet(args.output_size)
	elif args.type == "upsamplenet":
		build_upsamplenet(args.input_size, args.output_size)
	else:
		print(f"{args.type} is not a valid network type valid types are: {valid_network_types}")
		exit(1)

