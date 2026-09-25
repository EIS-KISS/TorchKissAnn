
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

"""Export a TorchKissAnn TorchScript network to ONNX.

The exported graph is pure network: it computes on the normalized input vector and returns the
network's own output space. All preprocessing that training applied to raw spectra (canonical
frequency resampling, per-spectra |Z| normalization, signed log compression, affine scaling) and
all postprocessing of the output (affine denormalization, exponentiation of log targets) is
described by metadata properties instead of being baked into the graph, so libkissinference can
apply it in its own domain types. Every property is optional; an absent property means that step
is the identity.

Metadata contract for the input side, applied in this order:

  canonicalGrid / canonicalFMin / canonicalFMax / canonicalPoints / canonicalLog
      resample the raw spectra onto that frequency grid (linear interpolation, constant edge
      extrapolation) before anything else
  spectraNorm
      'zmean', 'zgeom' or 'zmax': divide all 2N values by the spectra's mean |Z|, geometric mean
      |Z| or maximum |Z|
  inputLogCount / inputLogFloor
      apply sign(x) * log10(1 + |x| / inputLogFloor) to the first inputLogCount values
  inputScalars / inputBiases
      inputScalars[i] * x[i] + inputBiases[i]

And for the output side:

  outputScalars / outputBiases
      outputScalars[i] * y[i] + outputBiases[i]
  outputLog
      comma separated per-output flags; exponentiate (10^y) where the flag is 1

The network metadata is read from the meta.json extra file that buildnet.py embeds into the
TorchScript archive, so an exported model carries its full contract without side files.
"""

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


def addMetaProp(model, key: str, value) -> None:
	if checkCompositContainer(model.metadata_props, key):
		return
	prop = model.metadata_props.add()
	prop.key = key
	prop.value = str(value)


def addVectorMetaProp(model, key: str, values) -> None:
	if values is None:
		return
	addMetaProp(model, key, ",".join(str(v) for v in values))


def exportModel(model, meta, outname: str, output_prepend: str | None = None) -> str:
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

	# Output transform: the graph hands back values in the space the network was trained in, so the
	# client converts them to physical units with y*outputScalars + outputBiases, reading an entry
	# whose outputLog flag is set as the exponent of a base ten power (see the C++ affineToPhysical()).
	if 'outputBiases' in meta and len(meta['outputBiases']) == meta['outputSize']:
		addVectorMetaProp(onnx_model, 'outputBiases', meta['outputBiases'])

	if 'outputScalars' in meta and len(meta['outputScalars']) == meta['outputSize']:
		addVectorMetaProp(onnx_model, 'outputScalars', meta['outputScalars'])

	if meta.get('outputLog'):
		addVectorMetaProp(onnx_model, 'outputLog', [1 if v else 0 for v in meta['outputLog']])

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

	# Canonical frequency grid: the exported graph expects the already resampled 2N vector, so
	# libkissinference must resample a raw spectra onto this grid (linear interpolation, constant
	# edge extrapolation) before feeding it. Only present when the network was trained canonically.
	if meta.get('canonicalGrid'):
		for key in ('canonicalGrid', 'canonicalFMin', 'canonicalFMax', 'canonicalPoints', 'canonicalLog'):
			if key in meta and not checkCompositContainer(onnx_model.metadata_props, key):
				onnxmeta = onnx_model.metadata_props.add()
				onnxmeta.key = key
				onnxmeta.value = str(meta[key])

	# Input preparation contract. Nothing is baked into the graph any more: the exported network
	# computes on the normalized vector, so the client has to bring a raw spectra into that space
	# first, in exactly this order (the same order the training loaders use):
	#
	#   1. resample onto the canonical grid, if canonicalGrid is present
	#   2. divide all 2N values by the spectra's |Z| reference, if spectraNorm is present
	#      (zmean: mean of |Z|, zgeom: geometric mean, zmax: maximum of |Z|)
	#   3. signed log compression of the first inputLogCount values:
	#      sign(x) * log10(1 + |x| / inputLogFloor)
	#   4. inputScalars[i] * x[i] + inputBiases[i]
	#
	# Absent properties are identity at that step. Steps 3 and 4 are what regression training
	# derives from the dataset; classification only ever sets inputLogCount (the affine transform
	# would undo the per-spectra |Z| normalization).
	if meta.get('spectraNorm'):
		addMetaProp(onnx_model, 'spectraNorm', meta['spectraNorm'])
	if int(meta.get('inputLogCount', 0)) > 0:
		addMetaProp(onnx_model, 'inputLogCount', int(meta['inputLogCount']))
		addMetaProp(onnx_model, 'inputLogFloor', float(meta.get('inputLogFloor', 1e-9)))
	addVectorMetaProp(onnx_model, 'inputScalars', meta.get('inputScalars'))
	addVectorMetaProp(onnx_model, 'inputBiases', meta.get('inputBiases'))

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
