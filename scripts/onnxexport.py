import torch
import onnx
import sys
import json
import os
import onnxruntime as ort
from onnxruntime.training import artifacts as ortart
import argparse
import tarfile
import io

TMP_DIRECTORY = "./tmp"


def checkCompositContainer(container, key: str) -> bool:
	for element in container:
		if element.key == key:
			return True
	return False


def exportTrainingArtifactsModels(model_path: str, meta: dict, outname: str):
	onnx_model = onnx.load(model_path)

	requires_grad = [param.name for param in onnx_model.graph.initializer if "weight" in param.name or "bias" in param.name]

	out_dir = os.path.join(TMP_DIRECTORY, 'train_artifacts')
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	ortart.generate_artifacts(onnx_model, requires_grad=requires_grad, frozen_params=None,
	                          loss=ortart.LossType.CrossEntropyLoss, optimizer=ortart.OptimType.AdamW, artifact_directory=out_dir)

	try:
		os.remove(f"{outname}.tar")
	except FileNotFoundError:
		pass
	tar = tarfile.open(f"{outname}.tar", mode="x")
	for filename in os.listdir(out_dir):
		path = os.path.join(out_dir, filename)
		tar.add(path, arcname=os.path.split(path)[-1])
		os.remove(path)
	jsonstr = json.dumps(meta, indent='\t')
	jsonio = io.BytesIO(jsonstr.encode('utf-8'))
	jsontarinfo = tarfile.TarInfo("meta.json")
	jsontarinfo.size = len(jsonstr)
	tar.addfile(jsontarinfo, jsonio)
	tar.close()
	os.rmdir(out_dir)


def exportModel(model, meta, outname: str, train: bool = False, output_prepend: str | None = None) -> str:
	example_input = torch.randn((2 if train else 1, int(meta['inputSize'])), dtype=torch.float32)
	model.eval()
	example_output = model.forward(example_input)

	input_names = [meta['inputLabel'] if 'inputLabel' in meta is not None else "EIS"]
	output_names = [meta['purpose'] if 'purpose' in meta is not None else "Unkown"]
	if args.input_name is not None:
		input_names = [args.input_name]
	if args.purpose is not None:
		output_names = [args.purpose]

	torch.onnx.export(model, example_input, f"{outname}.onnx", verbose=True,
		training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
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
	parser.add_argument('--train', '-t', default=False, action="store_true", help="Also export files required to train onnx networks")
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
	exportModel(model, meta, basename, False, args.output_prepend)
	if args.train:
		print("Exporting tmp train model")
		exportModel(model, meta, basename + "_train", True, args.output_prepend)
		print("Exporting train model")
		exportTrainingArtifactsModels(f"{basename}_train.onnx", meta, basename + "_train")
		os.remove(f"{basename}_train.onnx")
