import torch
import onnx
import sys
import json
import os
import onnxruntime as ort
import argparse


if __name__ == "__main__":
	valid_network_types = "simple, conv, resnet"
	parser = argparse.ArgumentParser("A script to convert TorchScript networks trained by TorchKissAnn into onnx networks for libkissinference")
	parser.add_argument('--network', '-n', required=True, help="TorchScript network file name")
	parser.add_argument('--purpose', '-p', default=None, help="If set this will overide the purpose string of the network")
	parser.add_argument('--input_name', '-i', default=None, help="If set this will overide the input description string")
	parser.add_argument('--version', '-v', default=None, help="This will set the network version string to the given string")
	args = parser.parse_args()

	extra_files = {"meta.json": None}
	model = torch.jit.load(args.network, _extra_files=extra_files)
	if extra_files["meta.json"] is None:
		print("TorchScript dose not contain a meta.json")
		exit(1)

	basename = os.path.split(os.path.splitext(args.network)[0])[-1]

	meta = json.loads(extra_files["meta.json"])

	example_input = torch.randn((1, int(meta['inputSize'])), dtype=torch.float32)
	model.eval()
	example_output = model.forward(example_input)

	input_names = [meta['inputLabel'] if 'inputLabel' in meta is not None else "EIS"]
	output_names = [meta['purpose'] if 'purpose' in meta is not None else "Unkown"]
	if args.input_name is not None:
		input_names = [args.input_name]
	if args.purpose is not None:
		output_names = [args.purpose]

	torch.onnx.export(model, example_input, f"{basename}.onnx", verbose=True,
		training=torch.onnx.TrainingMode.EVAL, input_names=input_names, output_names=output_names)

	onnx_model = onnx.load(f"{basename}.onnx")
	onnx.checker.check_model(onnx_model)

	if 'outputLabels' in meta and len(meta['outputLabels']) == meta['outputSize']:
		onnxmeta = onnx_model.metadata_props.add()
		onnxmeta.key = 'outputLabels'
		value = ""
		for label in meta['outputLabels']:
			value = value + str(label) + ','
		onnxmeta.value = value[:-1]
	else:
		print("Warning: model lacks output labels")
		onnxmeta = onnx_model.metadata_props.add()
		onnxmeta.key = 'outputLabels'
		value = ""
		for i in range(0, meta['outputSize']):
			value = value + "class_" + str(i) + ','
		onnxmeta.value = value[:-1]

	onnxmeta = onnx_model.metadata_props.add()
	onnxmeta.key = "softmax"
	onnxmeta.value = str(meta['softmax'] if 'softmax' in meta is not None else "true")

	if args.version is not None:
		onnxmeta = onnx_model.metadata_props.add()
		onnxmeta.key = "version"
		onnxmeta.value = args.version

	onnx.save(onnx_model, f"{basename}.onnx")

	ort_model = ort.InferenceSession(f"{basename}.onnx")
	onnx_output = ort_model.run(None, {input_names[0]: example_input.numpy()})
	print(f"onnx output: {onnx_output}")
	print(f"torch output: {example_output}")
