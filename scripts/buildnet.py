
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
from nets.inception1d import InceptionTime1D
from nets.tcn1d import TCN1D
from nets.grunet import BiGRUNet
from nets.transformer1d import Transformer1D
from nets.seresnet1d import SEResNet1D
from nets.resnetv2_1d import ResNetV21D
from nets.decoder1d import DecoderNet
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


def build_resnet(output_size: int, base_filters: int = None):
	filters = 100 if base_filters is None else base_filters
	module = ResNet1D(in_channels=1,
		base_filters=filters,
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
	name = "resnet" if base_filters is None else f"resnet_f{filters}"
	save(script, None, output_size, name)


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


def build_inception(input_size: int, output_size: int, channels: int):
	module = InceptionTime1D(input_size, output_size, channels)

	data = torch.randn((16, input_size))
	data = module.forward(data)
	print(f"InceptionTime1D output size: {data.size()}")

	script = torch.jit.script(module)

	save(script, input_size, output_size, "inception")


def build_tcn(input_size: int, output_size: int, channels: int):
	module = TCN1D(input_size, output_size, channels)

	data = torch.randn((16, input_size))
	data = module.forward(data)
	print(f"TCN1D output size: {data.size()}")

	script = torch.jit.script(module)

	save(script, input_size, output_size, "tcn")


def build_gru(input_size: int, output_size: int, channels: int):
	module = BiGRUNet(input_size, output_size, channels)

	data = torch.randn((16, input_size))
	data = module.forward(data)
	print(f"BiGRUNet output size: {data.size()}")

	script = torch.jit.script(module)

	save(script, input_size, output_size, "gru")


def build_transformer(input_size: int, output_size: int, channels: int):
	module = Transformer1D(input_size, output_size, channels)

	data = torch.randn((16, input_size))
	data = module.forward(data)
	print(f"Transformer1D output size: {data.size()}")

	script = torch.jit.script(module)

	save(script, input_size, output_size, "transformer")


def build_seresnet(input_size: int, output_size: int, channels: int, extra_count: int = 0):
	module = SEResNet1D(input_size, output_size, channels, extra_count=extra_count)

	data = torch.randn((16, input_size))
	data = module.forward(data)
	print(f"SEResNet1D output size: {data.size()}")

	script = torch.jit.script(module)

	save(script, input_size, output_size, "seresnet")


def build_resnetv2(input_size: int, output_size: int, channels: int):
	module = ResNetV21D(input_size, output_size, channels)

	data = torch.randn((16, input_size))
	data = module.forward(data)
	print(f"ResNetV21D output size: {data.size()}")

	script = torch.jit.script(module)

	save(script, input_size, output_size, "resnetv2")


def build_decoder(input_size: int, output_size: int, channels: int):
	module = DecoderNet(input_size, output_size, channels)

	data = torch.randn((16, input_size))
	data = module.forward(data)
	print(f"DecoderNet output size: {data.size()}")

	script = torch.jit.script(module)

	save(script, input_size, output_size, "decoder")


if __name__ == "__main__":
	valid_network_types = "simple, conv, resnet, upsamplenet, inception, tcn, gru, transformer, seresnet, resnetv2, decoder"
	parser = argparse.ArgumentParser("TorchScript network generation script")
	parser.add_argument('--type', '-t', required=True, help=f"Type of network to create: {valid_network_types}")
	parser.add_argument('--output_size', '-o', type=int, required=True, help="Number of output classes")
	parser.add_argument('--input_size', '-i', type=int, default=100, help="Number of input classes")
	parser.add_argument('--channels', '-c', type=int, default=1,
		help="Input channels for the nets that support it (inception, tcn, gru, transformer, seresnet, resnetv2): "
			"1 keeps the whole spectra vector as a single channel like the older nets, "
			"2 splits it into an aligned real and imaginary channel (with --extra-inputs the extras are "
			"split off first, so 2 works for seresnet even on datasets with extra scalar inputs)")
	parser.add_argument('--extra-inputs', '-e', type=int, default=0,
		help="Number of trailing scalar extra inputs (temperature, ocv, ...) the loader appends behind the "
			"spectra; input_size must already include them. Only seresnet uses this: it builds a second "
			"branch (a small MLP) for those scalars that is fused with the conv trunk before the head. "
			"Ignored by the other network types")
	parser.add_argument('--base_filters', '-b', type=int, default=None,
		help="Base filter count of the resnet, 100 if omitted. The parameter count grows with the "
			"square of it and does not depend on the input size: at 8 outputs 24 filters are ~150k "
			"parameters and 34 are ~300k. When given, the count becomes part of the file name "
			"(resnet_f24ANY-8.pt) so several widths can be built into one directory")
	args = parser.parse_args()

	if args.extra_inputs > 0 and args.type != "seresnet":
		print(f"warning: --extra-inputs is only implemented for seresnet, ignoring it for {args.type}")

	if args.type == "simple":
		build_simplenet(args.input_size, args.output_size)
	elif args.type == "conv":
		build_convnet(args.input_size, args.output_size)
	elif args.type == "resnet":
		build_resnet(args.output_size, args.base_filters)
	elif args.type == "upsamplenet":
		build_upsamplenet(args.input_size, args.output_size)
	elif args.type == "inception":
		build_inception(args.input_size, args.output_size, args.channels)
	elif args.type == "tcn":
		build_tcn(args.input_size, args.output_size, args.channels)
	elif args.type == "gru":
		build_gru(args.input_size, args.output_size, args.channels)
	elif args.type == "transformer":
		build_transformer(args.input_size, args.output_size, args.channels)
	elif args.type == "seresnet":
		build_seresnet(args.input_size, args.output_size, args.channels, args.extra_inputs)
	elif args.type == "resnetv2":
		build_resnetv2(args.input_size, args.output_size, args.channels)
	elif args.type == "decoder":
		# a single feature map is too narrow to rebuild a spectra from a latent; default the
		# convolution width to 32 unless the caller asked for something explicit
		build_decoder(args.input_size, args.output_size, args.channels if args.channels > 1 else 32)
	else:
		print(f"{args.type} is not a valid network type valid types are: {valid_network_types}")
		exit(1)

