
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
import onnx
import sys
import json
import os
import onnxruntime as ort
import argparse
import tarfile
import io

TMP_DIRECTORY = "./tmp"


def checkCompositContainer(container, key: str) -> bool:
	for element in container:
		if element.key == key:
			return True
	return False


def vectorFromMeta(entry, size: int, default: float) -> torch.Tensor:
	if entry is None or len(entry) != size:
		return torch.full((size,), default, dtype=torch.float32)
	return torch.tensor([float(v) for v in entry], dtype=torch.float32)


class NormalizedModel(torch.nn.Module):
	"""
	Wrap a network that was trained on a normalized input and predicts normalized values, so the
	exported graph takes raw impedances and returns physical parameters.

	libkissinference only knows how to apply an affine transform to the output, so the log domains
	have to live inside the graph. The exported metadata therefore has to carry identity output
	scalars and biases, otherwise the conversion would be applied twice.
	"""

	def __init__(self, model, meta):
		super().__init__()
		self.model = model
		inputSize = int(meta['inputSize'])
		outputSize = int(meta['outputSize'])
		logCount = int(meta.get('inputLogCount', 0))
		self.logFloor = float(meta.get('inputLogFloor', 1e-9))
		logMask = torch.zeros(inputSize, dtype=torch.float32)
		logMask[:logCount] = 1.0
		self.register_buffer("logMask", logMask)
		self.register_buffer("inputScalars", vectorFromMeta(meta.get('inputScalars'), inputSize, 1.0))
		self.register_buffer("inputBiases", vectorFromMeta(meta.get('inputBiases'), inputSize, 0.0))
		self.register_buffer("outputScalars", vectorFromMeta(meta.get('outputScalars'), outputSize, 1.0))
		self.register_buffer("outputBiases", vectorFromMeta(meta.get('outputBiases'), outputSize, 0.0))
		self.register_buffer("outputLogMask", vectorFromMeta(
			[1.0 if v else 0.0 for v in meta['outputLog']] if meta.get('outputLog') else None, outputSize, 0.0))

	def forward(self, x):
		x = self.normalizeInput(x)
		return self.denormalizeOutput(self.model.forward(x))

	def normalizeInput(self, x):
		compressed = torch.sign(x)*torch.log10(1.0+torch.abs(x)/self.logFloor)
		x = torch.where(self.logMask > 0, compressed, x)
		return x*self.inputScalars + self.inputBiases

	def denormalizeOutput(self, y):
		y = y*self.outputScalars + self.outputBiases
		# clamped like the C++ side, so an overshooting network yields a huge number instead of inf
		return torch.where(self.outputLogMask > 0, torch.pow(10.0, torch.clamp(y, -30.0, 30.0)), y)


def isNormalized(meta) -> bool:
	return meta.get('inputScalars') is not None or bool(meta.get('outputLog'))


def exportModel(model, meta, outname: str, output_prepend: str | None = None) -> str:
	if isNormalized(meta):
		print("Baking the dataset normalization into the graph, output scalars become identity")
		model = NormalizedModel(model, meta)
		meta = dict(meta)
		meta['outputScalars'] = [1.0]*int(meta['outputSize'])
		meta['outputBiases'] = [0.0]*int(meta['outputSize'])

	example_input = torch.randn((1, int(meta['inputSize'])), dtype=torch.float32)
	model.eval()
	example_output = model.forward(example_input)

	input_names = [meta['inputLabel'] if 'inputLabel' in meta is not None else "EIS"]
	output_names = [meta['purpose'] if 'purpose' in meta is not None else "Unkown"]
	if args.input_name is not None:
		input_names = [args.input_name]
	if args.purpose is not None:
		output_names = [args.purpose]

	torch.onnx.export(model, example_input, f"{outname}.onnx", verbose=True,
		training=torch.onnx.TrainingMode.EVAL,
		input_names=input_names, output_names=output_names, do_constant_folding=False)

	onnx_model = onnx.load(f"{basename}.onnx")
	onnx.checker.check_model(onnx_model)

	if 'outputLabels' in meta and len(meta['outputLabels']) == meta['outputSize'] and not checkCompositContainer(onnx_model.metadata_props, 'outputLabels'):
		onnxmeta = onnx_model.metadata_props.add()
		onnxmeta.key = 'outputLabels'
		value = ""
		for label in meta['outputLabels']:
			label_str = str(label) if output_prepend is None else output_prepend + str(label)
			value = value + str(label_str) + ','
		onnxmeta.value = value[:-1]
	elif not checkCompositContainer(onnx_model.metadata_props, 'outputLabels'):
		print("Warning: model lacks output labels")
		onnxmeta = onnx_model.metadata_props.add()
		onnxmeta.key = 'outputLabels'
		value = ""
		for i in range(0, meta['outputSize']):
			value = value + "class_" + str(i) + ','
		onnxmeta.value = value[:-1]

	if 'outputBiases' in meta and len(meta['outputBiases']) == meta['outputSize'] and not checkCompositContainer(onnx_model.metadata_props, 'outputBiases'):
		onnxmeta = onnx_model.metadata_props.add()
		onnxmeta.key = 'outputBiases'
		biases = ""
		for bias in meta['outputBiases']:
			biases = biases + str(bias) + ','
		onnxmeta.value = biases[:-1]

	if 'outputScalars' in meta and len(meta['outputScalars']) == meta['outputSize'] and not checkCompositContainer(onnx_model.metadata_props, 'outputScalars'):
		onnxmeta = onnx_model.metadata_props.add()
		onnxmeta.key = 'outputScalars'
		scalars = ""
		for scalar in meta['outputScalars']:
			scalars = scalars + str(scalar) + ','
		onnxmeta.value = scalars[:-1]

	assert 'extraInputs' not in meta or 'extraInputLengths' in meta

	if 'extraInputs' in meta and not checkCompositContainer(onnx_model.metadata_props, 'extraInputs'):
		onnxmeta = onnx_model.metadata_props.add()
		onnxmeta.key = 'extraInputs'
		inputs = ""
		for einput in meta['extraInputs']:
			inputs = inputs + str(einput) + ','
		onnxmeta.value = inputs[:-1]

	if 'extraInputLengths' in meta and not checkCompositContainer(onnx_model.metadata_props, 'extraInputLengths'):
		onnxmeta = onnx_model.metadata_props.add()
		onnxmeta.key = 'extraInputLengths'
		inputs = ""
		for einput in meta['extraInputLengths']:
			inputs = inputs + str(einput) + ','
		onnxmeta.value = inputs[:-1]

	if not checkCompositContainer(onnx_model.metadata_props, 'softmax'):
		onnxmeta = onnx_model.metadata_props.add()
		onnxmeta.key = "softmax"
		onnxmeta.value = str(meta['softmax'] if 'softmax' in meta is not None else "true")

	if args.version is not None and not checkCompositContainer(onnx_model.metadata_props, 'version'):
		onnxmeta = onnx_model.metadata_props.add()
		onnxmeta.key = "version"
		onnxmeta.value = args.version

	onnx.save(onnx_model, f"{outname}.onnx")

	ort_model = ort.InferenceSession(f"{outname}.onnx")

	return "{outname}.onnx"


if __name__ == "__main__":
	valid_network_types = "simple, conv, resnet"
	parser = argparse.ArgumentParser("A script to convert TorchScript networks trained by TorchKissAnn into onnx networks for libkissinference")
	parser.add_argument('--network', '-n', required=True, help="TorchScript network file name")
	parser.add_argument('--purpose', '-p', default=None, help="If set this will overide the purpose string of the network")
	parser.add_argument('--input_name', '-i', default=None, help="If set this will overide the input description string")
	parser.add_argument('--version', '-v', default=None, help="This will set the network version string to the given string")
	parser.add_argument('--output_prepend', default=None, help="Optional string to prepend to output labels")
	args = parser.parse_args()

	extra_files = {"meta.json": None}
	model = torch.jit.load(args.network, _extra_files=extra_files)
	if extra_files["meta.json"] is None:
		print("TorchScript dose not contain a meta.json")
		exit(1)

	meta = json.loads(extra_files["meta.json"])

	basename = os.path.split(os.path.splitext(args.network)[0])[-1]
	print("Exporting eval model")
	exportModel(model, meta, basename, args.output_prepend)
