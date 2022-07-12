#!/bin/python

import torch as t

dummy_input = t.randn(100,1)
input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]

model = t.jit.load('./netBenchmark4.vos')
t.onnx.export(model , dummy_input, "mine.onnx", verbose=True, input_names=input_names, output_names=output_names)
