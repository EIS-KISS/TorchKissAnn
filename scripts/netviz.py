#!/usr/bin/env python3

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

"""Draw the structure of a TorchKissAnn network as a print friendly diagram.

Accepts either a TorchScript file (as produced by buildnet.py or by a training
run) or one of the network definitions in scripts/nets. The network is walked
through its TorchScript graph, executed layer by layer to measure the tensor
shapes, and reduced to a small number of labelled blocks. Output is monochrome
so it survives black and white printing.

Examples:
	./netviz.py -n resnetANY-100.pt -o resnet
	./netviz.py -t resnet -s 100 -o resnet -f pdf png
"""

import argparse
import importlib
import json
import os
import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import graphviz

warnings.filterwarnings("ignore")

# A scripted network refers back to the python it came from by qualified name,
# for example __torch__.nets.common.spectra_to_channels, so nets/ has to be
# importable no matter which directory netviz.py is started from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Layer classes that only ever decorate a real layer and can be folded into it.
NORMS = {"BatchNorm1d", "BatchNorm2d", "InstanceNorm1d", "GroupNorm", "LayerNorm"}
ACTIVATIONS = {"ReLU", "ReLU6", "LeakyReLU", "ELU", "GELU", "SiLU", "Sigmoid", "Tanh", "Mish"}
DROPOUTS = {"Dropout", "Dropout1d", "Dropout2d", "Dropout3d"}
DECOR = NORMS | ACTIVATIONS | DROPOUTS

# Layers that carry the meaning of a block, everything around them is folded in.
CORE = {"Conv1d", "Conv2d", "ConvTranspose1d", "Linear", "Bilinear"}

# Layers that are drawn whole rather than opened up. Their registered sub modules
# are parts of one computation, not a decomposition: an attention holds only its
# output projection as a child, so expanding it would draw a Linear where the
# attention is, and the diagram would no longer show attention at all.
ATOMIC = {"MultiheadAttention"}

# Class names that read better in a box than what torch calls them.
TITLES = {
	"NonDynamicallyQuantizableLinear": "Linear",
}

# aten ops that are noise in a block diagram, they are part of the layer they feed.
HIDDEN_OPS = {
	"aten::pad", "aten::constant_pad_nd", "aten::contiguous", "aten::transpose",
	"aten::permute", "aten::t", "aten::to", "aten::float", "aten::type_as",
	"aten::clone", "aten::detach", "aten::dropout", "aten::size", "aten::dim",
	"aten::__getitem__", "aten::ne", "aten::eq", "aten::__not__", "aten::floordiv",
	"aten::numel", "aten::Int", "aten::NumToTensor", "aten::scalarImplicit",
}

# aten ops that may be crossed when asking "does this value come from the input?".
# prim::CallMethod is in here because a shortcut often runs through a projection
# layer, add(out, projection(x)), which compiles to a call on that layer.
BYPASS_OPS = HIDDEN_OPS | {"prim::If", "prim::GetAttr", "prim::TupleConstruct",
	"prim::CallMethod"} | {
	"aten::max_pool1d", "aten::avg_pool1d", "aten::mean", "aten::sum"}

SKIP_OPS = {"aten::add", "aten::add_"}

# Arithmetic on shapes and flags. These run between the layers and have to be
# evaluated to work out the padding a layer needs, but they are not part of the
# data flow and never get a box.
SCALAR_OPS = {
	"aten::add": lambda a, b: a + b,
	"aten::sub": lambda a, b: a - b,
	"aten::mul": lambda a, b: a * b,
	"aten::div": lambda a, b: a / b,
	"aten::floordiv": lambda a, b: a // b,
	"aten::mod": lambda a, b: a % b,
	"aten::eq": lambda a, b: a == b,
	"aten::ne": lambda a, b: a != b,
	"aten::lt": lambda a, b: a < b,
	"aten::le": lambda a, b: a <= b,
	"aten::gt": lambda a, b: a > b,
	"aten::ge": lambda a, b: a >= b,
	"aten::__not__": lambda a: not a,
	"aten::Int": lambda a: int(a),
	"aten::__getitem__": lambda seq, index: seq[index],
	"aten::len": lambda seq: len(seq),
	# aten::size has two overloads and the bare call resolves to the wrong one,
	# so shape bookkeeping is evaluated here instead of through the dispatcher.
	"aten::size": lambda tensor, *dim: list(tensor.shape) if not dim else int(tensor.shape[dim[0]]),
	"aten::dim": lambda tensor: tensor.dim(),
	"aten::numel": lambda tensor: tensor.numel(),
	"prim::max": lambda *values: max(values),
	"prim::min": lambda *values: min(values),
	"prim::Not": lambda value: not value,
}

# Ops that only ever report on a tensor, they must not go through the dispatcher.
BOOKKEEPING_OPS = {"aten::size", "aten::dim", "aten::numel", "aten::__getitem__"}

# Free functions called through prim::CallFunction, mapped onto the aten op they run.
FUNCTION_KINDS = {
	"pad": "aten::pad",
	"relu": "aten::relu",
	"relu_": "aten::relu",
	"leaky_relu": "aten::leaky_relu",
	"dropout": "aten::dropout",
	"softmax": "aten::softmax",
	"linear": "aten::linear",
	"conv1d": "aten::conv1d",
	"cat": "aten::cat",
	"mean": "aten::mean",
	"max_pool1d": "aten::max_pool1d",
	"avg_pool1d": "aten::avg_pool1d",
	"interpolate": "aten::interpolate",
	"upsample": "aten::upsample",
	"batch_norm": "aten::batch_norm",
	"flatten": "aten::flatten",
	"sigmoid": "aten::sigmoid",
	"cross_entropy": "aten::cross_entropy",
	"_max_pool1d": "aten::max_pool1d",
	# The input reshaping helper of the nets in scripts/nets, drawn as the
	# reshape it is so the diagram shows where the channels come from.
	"spectra_to_channels": "aten::reshape",
}

# aten ops worth a box of their own when they are not part of a layer.
NOTABLE_OPS = {
	"aten::mean": "Global Average Pool",
	"aten::sum": "Sum",
	"aten::cat": "Concat",
	"aten::mul": "Multiply",
	"aten::max_pool1d": "MaxPool1d",
	"aten::avg_pool1d": "AvgPool1d",
	"aten::adaptive_avg_pool1d": "AdaptiveAvgPool1d",
	"aten::flatten": "Flatten",
	"aten::reshape": "Reshape",
	"aten::view": "Reshape",
	"aten::softmax": "Softmax",
	"aten::sigmoid": "Sigmoid",
	# Functional activations, as torch.nn.TransformerEncoderLayer and the
	# pre-activation blocks call them as plain ops instead of as modules. The
	# titles are the ones in ACTIVATIONS so they fold into the layer they follow.
	"aten::relu": "ReLU",
	"aten::relu_": "ReLU",
	"aten::relu6": "ReLU6",
	"aten::hardtanh": "ReLU6",
	"aten::leaky_relu": "LeakyReLU",
	"aten::elu": "ELU",
	"aten::gelu": "GELU",
	"aten::silu": "SiLU",
	"aten::mish": "Mish",
	"aten::tanh": "Tanh",
}

# A scripted loop is a real loop, so it is run as one. This is only a guard
# against a network whose trip count comes from the batch size.
MAX_LOOP_TRIPS = 64


class Unknown:
	"""Placeholder for a value the walker could not compute.

	__eq__ is defined so that membership tests never fall through to tensor
	comparison, which would return a tensor instead of a bool.
	"""

	def __repr__(self):
		return "?"

	def __eq__(self, other):
		return isinstance(other, Unknown)

	def __hash__(self):
		return hash("?")


UNKNOWN = Unknown()

# Marks an argument the caller of a module left out, distinct from a value the
# walker could not compute.
OMITTED = object()


def input_default(value):
	"""What a scripted forward sees for an argument the caller left out.

	The default lives in the schema rather than in the graph, so it has to come
	from the declared type: an optional mask is None, a flag is False. Getting
	this right matters, torch internals such as TransformerEncoder branch on it
	before they hand the tensor to their layers.
	"""
	declared = str(value.type())
	if declared.startswith("Optional"):
		return None
	if declared == "bool":
		return False
	if declared.startswith(("int", "float")):
		return 0
	return UNKNOWN


def clean_name(module) -> str:
	"""Human readable class name of a module, also for loaded TorchScript modules."""
	name = getattr(module, "original_name", None)
	if name:
		name = name.split(".")[-1]
	else:
		name = type(module).__name__
		if name == "RecursiveScriptModule":
			qualified = str(getattr(module, "qualified_name", "") or "")
			name = qualified.split(".")[-1].replace("___torch_mangle_0", "") or "Module"
	return TITLES.get(name, name)


def forward_methods(module) -> list:
	"""Names under which this module's forward can be looked up.

	A module scripted from a C++ builtin (nn.GRU and friends) stores its forward
	under a mangled name such as forward__0 and has no python attribute for it,
	so the compiled module has to be asked for its method names.
	"""
	names = ["forward"]
	if hasattr(module, "_c"):
		try:
			names += [n for n in module._c._method_names()
				if n.startswith("forward") and n not in names]
		except Exception:
			pass
	return names


def forward_graph(module):
	"""The graph of a module's forward, mangled names included."""
	for name in forward_methods(module):
		try:
			if name == "forward":
				return module.forward.graph
			return module._c._get_method(name).graph
		except Exception:
			continue
	raise AttributeError(f"{clean_name(module)} has no forward graph")


def method_graph(module, method: str):
	"""The graph of one named method of a scripted module."""
	if method == "forward":
		return forward_graph(module)
	return module._c._get_method(method).graph


def invoke_module(module, args):
	"""Call a module the way the graph does, mangled names included."""
	failure = None
	try:
		return module(*args)
	except AttributeError as error:
		failure = error
	if hasattr(module, "_c"):
		for name in forward_methods(module):
			if name == "forward":
				continue
			try:
				return module._c._get_method(name)(*args)
			except Exception as error:
				failure = error
	raise failure


def first_tensor(value):
	"""The tensor out of what a module returned, an RNN returns a pair."""
	if torch.is_tensor(value):
		return value
	if isinstance(value, (list, tuple)):
		for item in value:
			if torch.is_tensor(item):
				return item
	return None


def flatten_tensors(value) -> list:
	"""Every tensor in an argument, a list argument such as the one of cat included."""
	if torch.is_tensor(value):
		return [value]
	if isinstance(value, (list, tuple)):
		items = []
		for item in value:
			items.extend(flatten_tensors(item))
		return items
	return []


def human_count(value: int) -> str:
	if value >= 1_000_000:
		return f"{value / 1_000_000:.2f}M"
	if value >= 1_000:
		return f"{value / 1_000:.1f}k"
	return str(value)


def human_shape(shape) -> str:
	"""Format a shape with the batch dimension written as B, it is arbitrary."""
	if shape is None:
		return "?"
	dims = list(shape)
	if dims:
		dims[0] = "B"
	return " (" + ", ".join(str(d) for d in dims) + ")"


def pair(value) -> str:
	"""Render a one element tuple attribute such as kernel_size as a plain number."""
	if isinstance(value, (tuple, list)):
		if len(value) == 1:
			return str(value[0])
		return "x".join(str(v) for v in value)
	return str(value)


def describe(module) -> str:
	"""The parameters of a layer, the part that belongs in the box title."""
	cls = clean_name(module)
	try:
		if hasattr(module, "in_channels") and hasattr(module, "out_channels"):
			parts = [f"{module.in_channels} \u2192 {module.out_channels}"]
			if hasattr(module, "kernel_size"):
				parts.append(f"k{pair(module.kernel_size)}")
				if hasattr(module, "stride") and pair(module.stride) != "1":
					parts.append(f"s{pair(module.stride)}")
				if hasattr(module, "dilation") and pair(module.dilation) != "1":
					parts.append(f"d{pair(module.dilation)}")
			if hasattr(module, "groups") and int(module.groups) > 1:
				parts.append(f"g{int(module.groups)}")
			return " ".join(parts)
		if hasattr(module, "in_features") and hasattr(module, "out_features"):
			return f"{module.in_features} \u2192 {module.out_features}"
		if hasattr(module, "num_features"):
			return str(int(module.num_features))
		if hasattr(module, "num_channels"):
			return str(int(module.num_channels))
		if hasattr(module, "hidden_size"):
			parts = [f"h{int(module.hidden_size)}"]
			if int(getattr(module, "num_layers", 1)) > 1:
				parts.append(f"x{int(module.num_layers)}")
			if getattr(module, "bidirectional", False):
				parts.append("bi")
			return " ".join(parts)
		if hasattr(module, "normalized_shape"):
			shape = getattr(module, "normalized_shape")
			return pair(shape) if isinstance(shape, (tuple, list)) else str(shape)
		if cls.startswith(("MultiheadAttention", "TransformerEncoder", "TransformerDecoder")):
			parts = []
			# MultiheadAttention calls its width embed_dim/num_heads while the
			# transformer containers use d_model/nhead, so accept both spellings.
			width = getattr(module, "embed_dim", None)
			if width is None:
				width = getattr(module, "d_model", None)
			if width is not None:
				parts.append(str(int(width)))
			heads = getattr(module, "num_heads", None)
			if heads is None:
				heads = getattr(module, "nhead", None)
			if heads is not None:
				parts.append(f"{int(heads)} heads")
			if hasattr(module, "num_layers"):
				parts.append(f"x{int(module.num_layers)}")
			return " ".join(parts)
		if cls.startswith(("MaxPool", "AvgPool")) and hasattr(module, "kernel_size"):
			parts = [f"k{pair(module.kernel_size)}"]
			if hasattr(module, "stride") and pair(module.stride) != "1":
				parts.append(f"s{pair(module.stride)}")
			return " ".join(parts)
		if cls.startswith("Dropout"):
			return f"p={float(getattr(module, 'p', 0.5)):g}"
	except Exception:
		pass
	return ""


@dataclass
class Block:
	"""One box in the diagram, possibly with the internals of a layer group inside."""
	title: str
	detail: str = ""
	decor: list = field(default_factory=list)
	pre_decor: list = field(default_factory=list)
	in_shape: tuple = None
	out_shape: tuple = None
	params: int = 0
	residual: bool = False
	skip_via: str = ""
	children: list = field(default_factory=list)
	repeat: int = 1
	key: str = ""

	@property
	def label(self) -> str:
		core = self.title + (f"  {self.detail}" if self.detail else "")
		return " \u00b7 ".join(list(self.pre_decor) + [core] + list(self.decor))


class Capturer:
	"""Walks a TorchScript module and records what runs, in order, with shapes.

	TorchScript modules do not run python forward hooks and this build of torch
	does not fill in shapes during graph analysis, so the graph is interpreted
	node by node and every layer is executed to measure its shapes.

	The layers run with grad enabled, not under no_grad: the scripted
	MultiheadAttention aborts under no_grad, and with a one sample batch the
	autograd bookkeeping is nothing.
	"""

	def __init__(self, module):
		self.module = module
		self.warnings = []

	def walk(self, module, inputs, method="forward"):
		"""Execute module(inputs) and return (child blocks, output value)."""
		graph = method_graph(module, method)
		env = {}
		names = {}
		values = list(graph.inputs())
		env[values[0]] = module
		rest = values[1:]
		for index, value in enumerate(rest):
			if isinstance(inputs, (list, tuple)):
				given = inputs[index] if index < len(inputs) else OMITTED
			else:
				given = inputs if index == 0 else OMITTED
			env[value] = given if given is not OMITTED else input_default(value)
		blocks = self.exec_nodes(graph.nodes(), env, names, module, rest[0] if rest else None)
		out = UNKNOWN
		return_node = graph.return_node()
		if return_node is not None:
			returns = list(return_node.inputs())
			if returns:
				out = self.fetch(returns[0], env)
		return blocks, out

	def fetch(self, value, env):
		"""The value of a graph value, computing it from its inputs if needed.

		A TorchScript block may refer to a value defined outside of it, including
		one produced by a branch that was not taken, so a value can be missing
		where it is used. Only side effect free nodes are resolved that way.
		"""
		if value in env:
			return env[value]
		return self.evaluate(value, env, 0)

	def evaluate(self, value, env, depth):
		"""Best effort evaluation of a value the walk never reached."""
		if depth > 8:
			return UNKNOWN
		producer = value.node()
		if producer is None:
			return UNKNOWN
		kind = producer.kind()
		if kind == "prim::Param":
			return UNKNOWN
		if kind == "prim::Constant":
			try:
				result = producer.toIValue()
			except Exception:
				result = UNKNOWN
			env[value] = result
			return result
		args = [self.fetch(item, env) for item in list(producer.inputs())]
		result = UNKNOWN
		if UNKNOWN not in args:
			try:
				if kind == "prim::GetAttr":
					result = getattr(args[0], producer.s("name"))
				elif kind in ("prim::ListConstruct", "prim::TupleConstruct"):
					result = args
				elif kind in SCALAR_OPS:
					result = SCALAR_OPS[kind](*args)
				elif kind == "prim::CallFunction":
					function, _ = self.resolve_function(args[0])
					if function is not None:
						with torch.enable_grad():
							result = function(*args[1:])
			except Exception:
				result = UNKNOWN
		env[value] = result
		return result

	def exec_nodes(self, nodes, env, names, module, root) -> list:
		blocks = []
		for node in list(nodes):
			kind = node.kind()
			outputs = list(node.outputs())

			if kind == "prim::GetAttr":
				parent = list(node.inputs())[0]
				attribute = node.s("name")
				holder = env.get(parent)
				value = getattr(holder, attribute) if holder is not None else UNKNOWN
				if outputs:
					env[outputs[0]] = value
					if isinstance(value, torch.nn.Module):
						prefix = names.get(parent)
						names[outputs[0]] = attribute if prefix is None else f"{prefix}.{attribute}"
				continue

			if kind == "prim::Constant":
				if outputs:
					try:
						env[outputs[0]] = outputs[0].toIValue()
					except Exception:
						env[outputs[0]] = UNKNOWN
				continue

			if kind in ("prim::ListConstruct", "prim::TupleConstruct"):
				items = [self.fetch(v, env) for v in list(node.inputs())]
				if outputs and UNKNOWN not in items:
					env[outputs[0]] = items
				continue

			if kind in ("prim::TupleUnpack", "prim::ListUnpack"):
				# Split a composite such as the (output, h_n) pair an RNN returns.
				source = self.fetch(list(node.inputs())[0], env) if node.inputs() else UNKNOWN
				items = list(source) if isinstance(source, (list, tuple)) else None
				if items is not None and len(items) == len(outputs):
					for value, item in zip(outputs, items):
						env[value] = item
				else:
					for value in outputs:
						env[value] = UNKNOWN
				continue

			if kind == "prim::TupleIndex":
				inputs = list(node.inputs())
				source = self.fetch(inputs[0], env)
				index = self.fetch(inputs[1], env) if len(inputs) > 1 else UNKNOWN
				item = UNKNOWN
				if isinstance(source, (list, tuple)) and isinstance(index, int):
					if index < len(source):
						item = source[index]
				if outputs:
					env[outputs[0]] = item
				continue

			if kind == "prim::Uninitialized":
				# A slot for an optional value that is never filled in.
				for value in outputs:
					env[value] = None
				continue

			if kind == "prim::Loop":
				blocks.extend(self.run_loop(node, env, names, module, root))
				continue

			if kind == "prim::If":
				condition = self.fetch(list(node.inputs())[0], env)
				two_blocks = list(node.blocks())
				if not isinstance(condition, bool) or len(two_blocks) != 2:
					continue
				then_inputs = list(two_blocks[0].inputs())
				else_inputs = list(two_blocks[1].inputs())
				outer = list(node.inputs())[1:]
				# prim::If carries the then block captures first, then the else block captures.
				if condition:
					taken, sources = two_blocks[0], outer[:len(then_inputs)]
				else:
					taken, sources = two_blocks[1], outer[len(then_inputs):len(then_inputs) + len(else_inputs)]
				for inner, source in zip(list(taken.inputs()), sources):
					env[inner] = self.fetch(source, env)
				blocks.extend(self.exec_nodes(taken.nodes(), env, names, module, root))
				returned = taken.returnNode() if hasattr(taken, "returnNode") else None
				inner_out = list(returned.inputs()) if returned is not None else []
				for index, value in enumerate(outputs):
					if index < len(inner_out):
						env[value] = self.fetch(inner_out[index], env)
				continue

			if kind == "prim::CallMethod":
				ran = self.call_method(node, env, names, module)
				if isinstance(ran, list):
					blocks.extend(ran)
				elif ran is not None:
					blocks.append(ran)
				continue

			if kind == "prim::CallFunction":
				block = self.call_function(node, env)
				if block is not None:
					blocks.append(block)
				continue

			if kind == "aten::append":
				# A list the forward pass builds up item by item, keep it python.
				inputs = list(node.inputs())
				target = self.fetch(inputs[0], env) if inputs else UNKNOWN
				item = self.fetch(inputs[1], env) if len(inputs) > 1 else UNKNOWN
				if isinstance(target, list) and item is not UNKNOWN:
					target.append(item)
				for value in outputs:
					env[value] = target
				continue

			if kind.startswith("aten::") or kind.startswith("quantized::"):
				block = self.run_aten(node, env, kind)
				if block is not None:
					blocks.append(block)
				continue

			if kind in SCALAR_OPS:
				self.run_scalar(node, env, kind)
				continue
		return blocks

	def run_loop(self, node, env, names, module, root) -> list:
		"""Run a scripted loop, iterating a ModuleList compiles to this.

		The body is executed once per trip so every layer inside shows up in the
		diagram; the carried values are threaded through as the script does.
		"""
		inputs = list(node.inputs())
		outputs = list(node.outputs())
		blocks = []
		trips = self.fetch(inputs[0], env) if inputs else UNKNOWN
		if not isinstance(trips, int) or isinstance(trips, bool):
			trips = 1
		carried = [self.fetch(value, env) for value in inputs[2:]]
		for iteration in range(min(trips, MAX_LOOP_TRIPS)):
			for body in node.blocks():
				captures = [True, iteration] + carried
				for inner, source in zip(list(body.inputs()), captures):
					env[inner] = source
				blocks.extend(self.exec_nodes(body.nodes(), env, names, module, root))
				returned = body.returnNode() if hasattr(body, "returnNode") else None
				returns = list(returned.inputs()) if returned is not None else []
				if len(returns) == len(carried) + 1:
					carried = [self.fetch(value, env) for value in returns[1:]]
		for value, source in zip(outputs, carried):
			env[value] = source
		return blocks

	def resolve_function(self, value):
		"""Map a prim::CallFunction callee to something python can call."""
		producer = value.node()
		if producer is None:
			return None, ""
		name = ""
		module = ""
		try:
			name = producer.s("name")
		except Exception:
			name = ""
		if producer.kind() != "prim::GetAttr":
			# A constant function object, its type carries the qualified name,
			# for example __torch__.torch.nn.functional.pad or, for a helper
			# function from scripts/nets, __torch__.nets.common.spectra_to_channels
			qualified = str(value.type())
			if qualified.startswith("__torch__."):
				qualified = qualified[len("__torch__."):]
			if "." in qualified:
				module = qualified.rsplit(".", 1)[0]
				name = name or qualified.rsplit(".", 1)[-1]
		if not name:
			return None, ""
		if module:
			# Import the module the network was scripted from, this is also what
			# makes a custom layer function such as spectra_to_channels runnable.
			try:
				candidate = getattr(importlib.import_module(module), name, None)
				if callable(candidate):
					return candidate, FUNCTION_KINDS.get(name, f"aten::{name}")
			except Exception:
				pass
		for holder in (torch.nn.functional, torch):
			candidate = getattr(holder, name, None)
			if callable(candidate):
				return candidate, FUNCTION_KINDS.get(name, f"aten::{name}")
		return None, ""

	def call_function(self, node, env):
		"""Run a free function call such as torch.nn.functional.pad."""
		inputs = list(node.inputs())
		outputs = list(node.outputs())
		function, effective_kind = self.resolve_function(inputs[0])
		args = [self.fetch(v, env) for v in inputs[1:]]
		while args and args[-1] is UNKNOWN:
			# The caller passed something the walker could not work out, and the
			# module may still run if that argument is one it can do without.
			args.pop()
		result = UNKNOWN
		if function is not None and UNKNOWN not in args:
			try:
				with torch.enable_grad():
					result = function(*args)
			except Exception:
				result = UNKNOWN
			if result is UNKNOWN and effective_kind.startswith("aten::"):
				# The python helper takes arguments the dispatcher op does not,
				# so let the op itself try, with fewer of them if need be.
				try:
					result = self.invoke_aten(effective_kind, args)
				except Exception:
					result = UNKNOWN
		self.store(outputs, result, env)
		return self.record_op(effective_kind, args, result)

	@staticmethod
	def store(outputs, result, env):
		"""Record what an executed node produced.

		Only a node with several outputs returns something to split. A single
		output keeps its value whole, otherwise a list such as a shape would be
		spread over one output per element.
		"""
		if not outputs:
			return
		if len(outputs) == 1:
			env[outputs[0]] = result
		elif isinstance(result, (list, tuple)) and len(result) == len(outputs):
			for value, item in zip(outputs, result):
				env[value] = item
		else:
			for value in outputs:
				env[value] = UNKNOWN

	def call_method(self, node, env, names, parent) -> Block:
		inputs = list(node.inputs())
		outputs = list(node.outputs())
		try:
			method = node.s("name")
		except Exception:
			method = "forward"
		sub = env.get(inputs[0])
		if not isinstance(sub, torch.nn.Module):
			return None
		args = []
		for value in inputs[1:]:
			item = self.fetch(value, env)
			if item is UNKNOWN and str(value.type()).startswith("Optional"):
				# An optional argument the walk could not resolve, such as a mask
				# produced inside a branch that did not run: treat it as not given.
				item = None
			args.append(item)
		while args and args[-1] is UNKNOWN:
			# The caller passed something the walker could not work out, and the
			# module may still run if that argument is one it can do without.
			args.pop()
		if not method.startswith("forward"):
			# An internal helper method such as _sa_block. It is part of the data
			# path, so run it to keep the values flowing, and show what runs
			# inside it, but it is not a box itself: its receiver is the module.
			result = UNKNOWN
			try:
				with torch.enable_grad():
					result = sub._c._get_method(method)(*args)
			except Exception:
				result = UNKNOWN
			self.store(outputs, result, env)
			try:
				inner, _ = self.walk(sub, args, method)
			except Exception:
				inner = []
			return inner
		if clean_name(sub) == "Identity":
			# A placeholder for a layer this variant does not have, no box needed.
			self.store(outputs, args[0] if args else UNKNOWN, env)
			return None
		block, result = self.run_child(sub, args, names.get(inputs[0]))
		self.store(outputs, result, env)
		return block

	def run_child(self, sub, args, name=None) -> tuple:
		"""Execute one child module and turn it into a block with its internals."""
		shapes_in = tuple(tuple(a.shape) for a in args if torch.is_tensor(a))
		try:
			with torch.enable_grad():
				result = invoke_module(sub, args)
		except Exception as error:
			self.warnings.append(f"could not run {name or clean_name(sub)}: {error}")
			return None, UNKNOWN
		produced = first_tensor(result)
		block = Block(
			title=clean_name(sub),
			detail=describe(sub),
			in_shape=shapes_in[0] if shapes_in else None,
			out_shape=tuple(produced.shape) if produced is not None else None,
			params=sum(p.numel() for p in sub.parameters(recurse=True)),
			residual=False)

		# Recurse so that a custom layer shows what it is built from, except for
		# layers that are meant to be read as a single box.
		if len(list(sub.named_children())) > 0 and clean_name(sub) not in ATOMIC:
			children, _ = self.walk(sub, args if len(args) > 1 else (args[0] if args else None))
			if not children:
				children = self.walk_children(sub, args)
			block.children = children
			residual, via = detect_skip(sub)
			block.residual = residual
			block.skip_via = via
		return block, result

	def walk_children(self, sub, args) -> list:
		"""Run the children of a module in registration order.

		Fallback for modules whose forward is not a call graph worth walking:
		torch internals such as TransformerEncoder spend it on mask and device
		bookkeeping and only hand the tensor to their layers at the very end.
		"""
		blocks = []
		value = args[0] if args else UNKNOWN
		for name, child in sub.named_children():
			block, result = self.run_child(child, [value], name)
			if block is None:
				return []
			blocks.append(block)
			produced = first_tensor(result)
			if produced is not None:
				value = produced
		return blocks

	def invoke_aten(self, kind, args):
		"""Call an aten op with the arguments the graph node carries.

		A graph node lists every argument including the optional ones, while the
		python overloads accept fewer, so trailing arguments are dropped until
		one of the calls matches.
		"""
		operation = getattr(torch.ops.aten, kind.split("::", 1)[1])
		failure = None
		for count in range(len(args), 0, -1):
			try:
				with torch.enable_grad():
					return operation(*args[:count])
			except Exception as error:
				failure = error
		raise failure

	def run_aten(self, node, env, kind):
		inputs = list(node.inputs())
		outputs = list(node.outputs())
		args = [self.fetch(v, env) for v in inputs]
		tensors = [item for arg in args for item in flatten_tensors(arg)]
		result = UNKNOWN
		if UNKNOWN not in args:
			try:
				if kind in SCALAR_OPS and (kind in BOOKKEEPING_OPS or not tensors):
					result = SCALAR_OPS[kind](*args)
				elif tensors:
					result = self.invoke_aten(kind, args)
				elif kind in SCALAR_OPS:
					result = SCALAR_OPS[kind](*args)
			except Exception:
				result = UNKNOWN
		self.store(outputs, result, env)
		return self.record_op(kind, args, result)

	def run_scalar(self, node, env, kind):
		"""Evaluate shape arithmetic such as prim::max, it never gets a box."""
		outputs = list(node.outputs())
		args = [self.fetch(v, env) for v in list(node.inputs())]
		result = UNKNOWN
		if UNKNOWN not in args:
			try:
				result = SCALAR_OPS[kind](*args)
			except Exception:
				result = UNKNOWN
		self.store(outputs, result, env)
		return None

	def record_op(self, kind, args, result):
		"""Decide whether an executed op deserves a box of its own."""
		tensors = [item for arg in args for item in flatten_tensors(arg)]
		if kind in SKIP_OPS and tensors:
			# The result, not the last input: adding a broadcast positional
			# embedding would otherwise report the shape of the embedding.
			tensor_out = result if torch.is_tensor(result) else None
			if tensor_out is None:
				tensor_out = tensors[-1]
			return Block(title="Add", in_shape=tuple(t.shape for t in tensors),
				out_shape=tuple(tensor_out.shape))
		if kind in HIDDEN_OPS or not tensors:
			return None
		name = NOTABLE_OPS.get(kind)
		if name is None:
			return None
		tensor_out = result if torch.is_tensor(result) else None
		if tensor_out is None:
			return None
		return Block(title=name, in_shape=tuple(tensors[0].shape), out_shape=tuple(tensor_out.shape))


def detect_skip(module):
	"""Does this module add its own input back onto its output, and through what?"""
	try:
		graph = forward_graph(module)
	except Exception:
		return False, ""
	via = []
	found = skip_in_nodes(graph.nodes(), module, via)
	return found, " + ".join(dict.fromkeys(via))


def skip_in_nodes(nodes, module, via) -> bool:
	"""Does an add in these nodes, or in a branch of them, close a shortcut?

	A branch whose condition was already decided at script time and that did not
	run is not part of the network, which is what keeps an optional residual
	connection from being reported in the variant that switched it off.
	"""
	found = False
	for node in nodes:
		if node.kind() in SKIP_OPS:
			for value in list(node.inputs()):
				if not isinstance(value.type(), torch.TensorType):
					continue
				chain = skip_source(value, module)
				if chain is not None:
					via.extend(chain)
					found = True
		blocks = branch_blocks(node, module)
		for block in blocks:
			if skip_in_nodes(list(block.nodes()), module, via):
				found = True
	return found


def static_value(value, module):
	"""What a graph value holds when it was fixed at script time, else None."""
	producer = value.node()
	if producer is None:
		return None
	kind = producer.kind()
	if kind == "prim::Param":
		return module
	if kind == "prim::Constant":
		try:
			return producer.toIValue()
		except Exception:
			return None
	if kind == "prim::GetAttr":
		parent = static_value(list(producer.inputs())[0], module)
		if not isinstance(parent, torch.nn.Module):
			return None
		try:
			return getattr(parent, producer.s("name"))
		except Exception:
			return None
	return None


def skip_source(value, module, depth: int = 0):
	"""Trace a value back over the ops a shortcut may run through.

	Returns what it passed through once it lands on one of the module's own
	inputs, which is what a residual connection is, and None when it does not.
	"""
	chain = []
	current = value
	for _ in range(12):
		producer = current.node()
		if producer is None:
			return None
		kind = producer.kind()
		if kind == "prim::Param":
			return chain
		if kind == "prim::If":
			# The value comes out of a branch, so follow what the branches
			# return; a branch decided at script time is not part of the network.
			if depth > 4:
				return None
			blocks = branch_blocks(producer, module)
			try:
				index = list(producer.outputs()).index(current)
			except ValueError:
				index = 0
			for block in blocks:
				returned = block.returnNode()
				if returned is None:
					continue
				returns = list(returned.inputs())
				if index >= len(returns):
					continue
				inner = skip_source(returns[index], module, depth + 1)
				if inner is not None:
					return chain + inner
			return None
		if kind not in BYPASS_OPS:
			return None
		if kind in ("aten::max_pool1d", "aten::avg_pool1d"):
			chain.append("pool")
		# The first tensor input: a branch node lists its condition first and a
		# module call lists the module itself first.
		inner = [item for item in producer.inputs()
			if isinstance(item.type(), torch.TensorType)]
		if not inner:
			return None
		current = inner[0]
	return None


def branch_blocks(node, module) -> list:
	"""The blocks of a branch node that can still run for this network."""
	blocks = list(node.blocks())
	if node.kind() != "prim::If" or len(blocks) != 2:
		return blocks
	condition = static_value(list(node.inputs())[0], module)
	if isinstance(condition, bool):
		return [blocks[0] if condition else blocks[1]]
	return blocks


CONTAINERS = {"Sequential", "ModuleList", "ModuleDict"}


def expand_containers(blocks: list) -> list:
	"""Splice Sequential and friends away, they group layers nobody designed."""
	out = []
	for block in blocks:
		if block.title in CONTAINERS and not block.residual:
			out.extend(expand_containers(block.children))
		else:
			out.append(block)
	return out


def simplify(blocks: list, pre: bool = False, collapse: bool = True) -> list:
	"""Turn a faithful op list into the few boxes that carry the design.

	Each level is processed exactly once: containers are spliced away before
	the recursion, so their children are simplified and collapsed together with
	their new siblings. `pre` says the group is pre-activation, which is true
	inside a block with a shortcut, where norm and activation precede their
	layer. Runs are not collapsed inside a group with a shortcut, because a
	group reduced to a single box would leave the shortcut arrow nothing to
	skip over.
	"""
	clean = []
	for block in expand_containers(blocks):
		block.children = simplify(block.children, block.residual, not block.residual)
		# A wrapper that only holds a single real layer adds nothing, e.g. Conv1dSamePad.
		while block.children and len(block.children) == 1 and block.params == block.children[0].params:
			inner = block.children[0]
			inner.residual = inner.residual or block.residual
			inner.skip_via = inner.skip_via or block.skip_via
			inner.in_shape = block.in_shape or inner.in_shape  # the wrapper is invisible
			block = inner
		clean.append(block)
	clean = fold_decor(clean, pre)
	for block in clean:
		tidy_residual(block)
	return collapse_repeats(clean) if collapse else clean


def fold_decor(blocks: list, force_pre: bool = False) -> list:
	"""Fold batchnorm, activation and dropout layers into the layer they belong to.

	A group that opens with norm or activation is pre-activation, like the
	residual blocks in these networks, and every such run then belongs to the
	layer that follows it. Otherwise a run belongs to the layer before it.
	"""
	if not blocks:
		return blocks

	pre = force_pre or (blocks[0].title in DECOR and not blocks[0].children)
	merged = []
	pending = []

	def own_box(run):
		titles = [item.title for item in run]
		box = run[0]
		box.title = "Norm / Activation"
		box.detail = ""
		box.decor = titles
		box.params = sum(item.params for item in run)
		box.out_shape = run[-1].out_shape
		merged.append(box)

	def forward(target):
		target.pre_decor.extend(item.title for item in pending)
		target.params += sum(item.params for item in pending)
		pending.clear()

	def backward(target):
		target.decor.extend(item.title for item in pending)
		target.params += sum(item.params for item in pending)
		target.out_shape = pending[-1].out_shape or target.out_shape
		pending.clear()

	for block in blocks:
		if block.title in DECOR and not block.children:
			pending.append(block)
			continue
		if pending:
			if pre and not block.children:
				forward(block)
			elif not pre and merged and not merged[-1].children:
				backward(merged[-1])
			else:
				own_box(pending)
				pending.clear()
		merged.append(block)
	if pending:
		if merged and not merged[-1].children:
			backward(merged[-1])
		else:
			own_box(pending)
	return merged


POOL_TITLES = {"MaxPool1d", "AvgPool1d", "AdaptiveAvgPool1d", "MaxPoolSamePad", "Pool1dBlock"}


def tidy_residual(block: Block):
	"""The shortcut arrow already shows the identity path, so it needs no boxes."""
	if not block.residual or not block.children:
		return
	last_core = -1
	for index, child in enumerate(block.children):
		if not child.children and child.title != "Add" and child.title not in POOL_TITLES:
			last_core = index
	kept = []
	for index, child in enumerate(block.children):
		if child.title == "Add":
			continue
		if index > last_core and child.title in POOL_TITLES:
			continue
		kept.append(child)
	block.children = kept


def signature(block: Block) -> str:
	return "|".join([block.title, block.detail, str(block.in_shape), str(block.out_shape),
		",".join(block.pre_decor), ",".join(block.decor), str(block.residual),
		",".join(signature(c) for c in block.children)])


def collapse_repeats(blocks: list) -> list:
	"""Merge runs of identical consecutive blocks into one box marked x N."""
	out = []
	index = 0
	while index < len(blocks):
		block = blocks[index]
		count = 1
		while index + count < len(blocks) and signature(blocks[index + count]) == signature(block):
			count += 1
		block.repeat = count
		out.append(block)
		index += count
	return out


def flatten_single(blocks: list) -> list:
	"""Drop boxes that only wrap a single other box, they cost space and say nothing."""
	out = []
	for block in blocks:
		block.children = flatten_single(block.children)
		while block.children and len(block.children) == 1 and not block.residual:
			inner = block.children[0]
			if inner.residual or inner.decor:
				break
			if block.params != inner.params:
				# The child is only a part of the box rather than everything in
				# it, so unwrapping would rename the layer to one of its parts.
				break
			inner.in_shape = block.in_shape or inner.in_shape
			block = inner
			block.children = flatten_single(block.children)
		out.append(block)
	return out


class Renderer:
	"""Draws the block tree as a monochrome graph, laid out top to bottom.

	Everything is black on white or on light gray so the diagram survives
	being printed in black and white.
	"""

	def __init__(self):
		self.digraph = graphviz.Digraph(graph_attr={
			"rankdir": "TB",
			"newrank": "true",
			"bgcolor": "white",
			"nodesep": "0.24",
			"ranksep": "0.38",
			"margin": "0.4",
			"labelloc": "t",
			"labeljust": "c",
			"fontname": "Helvetica-Bold",
			"fontsize": "12",
			"fontcolor": "black",
			"compound": "true",
		})
		self.counter = 0

	def new_id(self, prefix: str) -> str:
		self.counter += 1
		return f"{prefix}{self.counter}"

	def box(self, target, block: Block) -> str:
		"""Draw a block without internals as a rounded white box."""
		identifier = self.new_id("n")
		label = block.label
		if block.params:
			label += f"\n{human_count(block.params)} params"
		target.node(identifier, label=label, shape="box", style="rounded,filled",
			fillcolor="white", color="black", penwidth="1.1", fontname="Helvetica",
			fontsize="10", margin="0.16,0.09")
		return identifier

	def plus(self, target) -> str:
		"""The node a shortcut is merged back into."""
		identifier = self.new_id("plus")
		target.node(identifier, label="+", shape="circle", style="solid", color="black",
			fontname="Helvetica-Bold", fontsize="12", width="0.3", fixedsize="true")
		return identifier

	def link(self, target, source: str, destination: str, dashed: bool = False, label: str = ""):
		target.edge(source, destination, color="black", penwidth="1.1", arrowsize="0.7",
			style="dashed" if dashed else "solid", fontsize="9",
			fontname="Helvetica-Oblique", fontcolor="black", label=label)

	def group(self, parent, block: Block) -> tuple:
		"""Draw a block with internals as a gray box around its children."""
		cluster = graphviz.Digraph(name=self.new_id("cluster_"), graph_attr={
			"style": "rounded",
			"color": "black",
			"penwidth": "1.0",
			"bgcolor": "gray93",
			"labeljust": "l",
			"margin": "30",
			"fontname": "Helvetica-bold",
			"fontsize": "10",
		})
		title = block.title + (f" {block.detail}" if block.detail else "")
		if block.params:
			title += f"\n{human_count(block.params)} params"
		cluster.attr(label=title)

		entry = None
		exit_node = None
		for child in block.children:
			child_entry, child_exit = self.draw(cluster, child)
			if entry is None:
				entry = child_entry
			if exit_node is not None:
				self.link(cluster, exit_node, child_entry)
			exit_node = child_exit

		if block.residual:
			merged = self.plus(cluster)
			self.link(cluster, exit_node, merged)
			via = f"shortcut ({block.skip_via})" if block.skip_via else "shortcut"
			if entry == exit_node:
				# With a single child the shortcut would retrace the main path
				# edge and skip nothing, so branch it off a junction above it.
				branch = self.new_id("branch")
				cluster.node(branch, label="", shape="point", color="black", width="0.02")
				self.link(cluster, branch, entry)
				self.link(cluster, branch, merged, dashed=True, label=via)
				entry = branch
			else:
				self.link(cluster, entry, merged, dashed=True, label=via)
			exit_node = merged

		parent.subgraph(cluster)
		return entry, exit_node

	def draw(self, target, block: Block) -> tuple:
		"""Draw one block, return the (entry, exit) nodes to wire it into a chain."""
		if block.repeat > 1:
			return self.draw_repeated(target, block)
		if not block.children:
			identifier = self.box(target, block)
			return identifier, identifier
		return self.group(target, block)

	def draw_repeated(self, target, block: Block) -> tuple:
		"""Show a repeated block as several copies in a row.

		Up to three copies are drawn whole, more than that and the first and
		the last copy bracket a dots node carrying the total count.
		"""
		count = block.repeat
		block.repeat = 1
		entry = None
		previous = None
		for index in range(count if count <= 3 else 2):
			if index == 1 and count > 3:
				dots = self.dots(target, count)
				self.link(target, previous, dots)
				previous = dots
			copy_entry, copy_exit = self.draw(target, block)
			if entry is None:
				entry = copy_entry
			if previous is not None:
				self.link(target, previous, copy_entry)
			previous = copy_exit
		block.repeat = count
		return entry, previous

	def dots(self, target, count: int) -> str:
		"""The placeholder standing in for the copies between the outer two."""
		identifier = self.new_id("dots")
		target.node(identifier, label=f"\u22ee\n\u00d7{count}", shape="none",
			fontname="Helvetica-Bold", fontsize="12", fontcolor="black")
		return identifier

	def chain(self, blocks: list, input_shape: str, output_shape: str):
		start = self.new_id("in")
		self.digraph.node(start, label=f"Input\n{input_shape}", shape="oval", style="bold",
			color="black", penwidth="1.4", fontname="Helvetica-Bold", fontsize="11")
		previous = start
		for block in blocks:
			entry, exit_node = self.draw(self.digraph, block)
			shape = human_shape(block.in_shape) if block.in_shape else ""
			self.digraph.edge(previous, entry, label=shape, color="black", penwidth="1.1",
				arrowsize="0.7", fontsize="9", fontname="Helvetica", fontcolor="black")
			previous = exit_node
		end = self.new_id("out")
		self.digraph.node(end, label=f"Output\n{output_shape}", shape="oval", style="bold",
			color="black", penwidth="1.4", fontname="Helvetica-Bold", fontsize="11")
		self.digraph.edge(previous, end, color="black", penwidth="1.1", arrowsize="0.7")


@contextmanager
def fused_mha_off():
	"""Turn off the fused transformer fast paths while the network is walked.

	With them enabled torch replaces a whole encoder layer by a single aten
	call, which leaves nothing to draw. Shapes are the same either way.
	"""
	try:
		previous = torch.backends.mha.get_fastpath_enabled()
		torch.backends.mha.set_fastpath_enabled(False)
	except Exception:
		previous = None
	try:
		yield
	finally:
		if previous is not None:
			try:
				torch.backends.mha.set_fastpath_enabled(previous)
			except Exception:
				pass


def load_torchscript(path: str):
	extra = {"meta.json": None}
	module = torch.jit.load(path, _extra_files=extra)
	module.eval()
	meta = {}
	if extra["meta.json"] is not None:
		try:
			meta = json.loads(extra["meta.json"].decode())
		except Exception:
			meta = {}
	return module, meta


def build_source(kind: str, input_size: int, output_size: int, channels: int = 1):
	"""Instantiate one of the networks in scripts/nets, as buildnet.py does."""
	if kind == "simple":
		from nets.simplenet import SimpleNet
		module = SimpleNet(input_size, output_size, 4, 3)
	elif kind == "conv":
		from nets.convnet import ConvNet
		module = ConvNet(input_size, output_size, 4, 3)
	elif kind == "resnet":
		from nets.resnet1d import ResNet1D
		module = ResNet1D(in_channels=1, base_filters=100, kernel_size=16, stride=2, groups=1,
			n_classes=output_size, downsample_gap=output_size, increasefilter_gap=12)
	elif kind == "upsamplenet":
		from nets.upsamplenet import UpsampleNet
		module = UpsampleNet(input_size, output_size, 4, 3)
	elif kind == "inception":
		from nets.inception1d import InceptionTime1D
		module = InceptionTime1D(input_size, output_size, channels)
	elif kind == "tcn":
		from nets.tcn1d import TCN1D
		module = TCN1D(input_size, output_size, channels)
	elif kind == "gru":
		from nets.grunet import BiGRUNet
		module = BiGRUNet(input_size, output_size, channels)
	elif kind == "transformer":
		from nets.transformer1d import Transformer1D
		module = Transformer1D(input_size, output_size, channels)
	elif kind == "seresnet":
		from nets.seresnet1d import SEResNet1D
		module = SEResNet1D(input_size, output_size, channels)
	else:
		raise SystemExit(f"unknown network type {kind}")
	module.eval()
	return torch.jit.script(module)


def dump_tree(blocks: list, indent: int = 0):
	"""Print the block tree, useful to check what the walker recognised."""
	for block in blocks:
		head = " " * indent + block.title
		if block.detail:
			head += f"  {block.detail}"
		if block.pre_decor or block.decor:
			head += " [" + " ".join(list(block.pre_decor) + list(block.decor)) + "]"
		if block.repeat > 1:
			head += f"{block.repeat}x"
		if block.residual:
			head += "  +shortcut"
		head += f" -> {human_shape(block.out_shape)}"
		if block.params:
			head += f"({human_count(block.params)})"
		print(head)
		dump_tree(block.children, indent + 1)


def main():
	parser = argparse.ArgumentParser(
		"TorchKissAnn network structure visualiser, draws a network as a monochrome block diagram")
	source = parser.add_mutually_exclusive_group(required=True)
	source.add_argument("--network", "-n", help="TorchScript network file, e.g. resnetANY-100.pt")
	source.add_argument("--type", "-t",
		choices=["simple", "conv", "resnet", "upsamplenet",
			"inception", "tcn", "gru", "transformer", "seresnet"],
		help="Build a network from scripts/nets instead of reading a file")
	parser.add_argument("--input_size", "-i", type=int, default=100, help="Input length for --type")
	parser.add_argument("--output_size", "-s", type=int, default=10, help="Output classes for --type")
	parser.add_argument("--channels", "-c", type=int, default=1,
		help="Input channels for --type of the nets that support it: 1 is the whole vector as one"
			" channel, 2 splits it into an aligned real and imaginary channel pair")
	parser.add_argument("--out", "-o", default=None, help="Output file name without extension")
	parser.add_argument("--format", "-f", nargs="+", default=["pdf"],
		choices=["pdf", "svg", "png"], help="Output formats, pdf is best for print")
	parser.add_argument("--batch", type=int, default=1, help="Batch size used to measure shapes")
	parser.add_argument("--keep-gv", action="store_true", help="Also write the graphviz source")
	parser.add_argument("--dump", action="store_true",
		help="Print the recognised block tree as text instead of drawing it")
	args = parser.parse_args()

	if args.network:
		module, meta = load_torchscript(args.network)
		name = meta.get("name") or os.path.splitext(os.path.basename(args.network))[0]
		input_size = meta.get("inputSize") or args.input_size
		output_size = meta.get("outputSize") or args.output_size
	else:
		module = build_source(args.type, args.input_size, args.output_size, args.channels)
		meta = {"name": args.type}
		name = args.type
		input_size = args.input_size
		output_size = args.output_size

	input_size = int(input_size)
	sample = torch.randn((args.batch, input_size))
	with torch.enable_grad():
		final = module(sample)

	with fused_mha_off():
		capturer = Capturer(module)
		blocks, _ = capturer.walk(module, sample)
	blocks = flatten_single(simplify(blocks))

	for warning in capturer.warnings:
		print(f"warning: {warning}", file=sys.stderr)

	if args.dump:
		dump_tree(blocks)
		return

	total = sum(p.numel() for p in module.parameters())
	out = args.out or (f"{name}_{input_size}-{output_size}" if args.network else name)

	renderer = Renderer()
	renderer.chain(blocks, human_shape((args.batch, input_size)),
		human_shape(tuple(final.shape) if torch.is_tensor(final) else (args.batch, output_size)))

	for fmt in args.format:
		path = renderer.digraph.render(filename=out, format=fmt, cleanup=True)
		print(f"wrote {path}")
	if args.keep_gv:
		source = f"{out}.gv"
		with open(source, "w", encoding="utf-8") as handle:
			handle.write(renderer.digraph.source)
		print(f"wrote {source}")


if __name__ == "__main__":
	main()
