//
// TorchKissAnn - A collection of tools to train various types of Machine learning
// algorithms on various types of EIS data
// Copyright (C) 2025 Carl Klemm <carl@uvos.xyz>
//
// This file is part of TorchKissAnn.
//
// TorchKissAnn is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// TorchKissAnn is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with TorchKissAnn.  If not, see <http://www.gnu.org/licenses/>.
//

#include "convnet.h"
#include "../utils/seqprint.h"
#include <sstream>
#include <fstream>

using namespace ann;

static std::pair<size_t, size_t> layerWidthFunction(size_t layer, size_t leyers, size_t input, size_t output)
{
	size_t layerIn = (input*(leyers-layer)/leyers)+output*(layer)/leyers;
	size_t layerOut = (input*(leyers-layer-1)/leyers)+output*(layer+1)/leyers;
	return std::pair<size_t, size_t>(layerIn, layerOut);
}

ann::ConvNet::ConvNet(const Json::Value& node):
Net(node)
{
	downsampleSteps = node["downsampleSteps"].asUInt64();
	extraSteps = node["extraSteps"].asUInt64();
	init();
}

ann::ConvNet::ConvNet(int64_t inputSizeI, int64_t outputSizeI, size_t downsampleStepsI, size_t extraStepsI, bool softmaxI):
Net(inputSizeI, outputSizeI, softmaxI), downsampleSteps(downsampleStepsI), extraSteps(extraStepsI)
{
	init();
}

void ann::ConvNet::init()
{
	int64_t size = inputSize;
	const int conv1Kernel = inputSize/5;
	const int maxPool1Devisor = 2;
	const int conv2Kernel = inputSize/10;
	const int maxPool2Devisor = 2;

	register_module("ConvModel", model);

	//conv part
	//model->push_back(SeqPrint());
	torch::nn::Conv1d conv1d0(torch::nn::Conv1dOptions(1, 3, conv1Kernel).stride(1).bias(false));
	model->push_back(conv1d0);
	size = size-conv1Kernel+1;
	//model->push_back(SeqPrint());
	model->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)));
	model->push_back(torch::nn::MaxPool1d(torch::nn::MaxPool1dOptions(maxPool1Devisor)));
	size = size/maxPool1Devisor;
	//model->push_back(SeqPrint());
	model->push_back(torch::nn::Conv1d(torch::nn::Conv1dOptions(3, 6, conv2Kernel).stride(1).bias(false)));
	size = size-conv2Kernel+1;
	//model->push_back(SeqPrint());
	model->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)));
	model->push_back(torch::nn::MaxPool1d(torch::nn::MaxPool1dOptions(maxPool2Devisor)));
	size = size/2;
	//model->push_back(SeqPrint());
	model->push_back(torch::nn::Flatten());
	size = size*6;
	//model->push_back(SeqPrint());

	//fully connected layers
	std::pair<size_t, size_t> sizes;
	for(size_t i = 0; i < downsampleSteps-1; ++i)
	{
		sizes = layerWidthFunction(i, downsampleSteps, size, outputSize);
		model->push_back(torch::nn::Linear(sizes.first, sizes.second));
		model->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)));
		torch::nn::BatchNorm1d bn(torch::nn::BatchNorm1dOptions(sizes.second).eps(0.5).momentum(0.1).affine(false));
		model->push_back(bn);

		if(i == 0)
		{
			for(size_t j = 0; j < extraSteps; ++j)
			{
				model->push_back(torch::nn::Linear(sizes.second, sizes.second));
				model->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)));
			}
		}
	}

	sizes = layerWidthFunction(downsampleSteps-1, downsampleSteps, size, outputSize);
	model->push_back(torch::nn::Linear(sizes.first, sizes.second));
	if(softmax)
		model->push_back(torch::nn::LogSoftmax(torch::nn::LogSoftmaxOptions(1)));
}

torch::Tensor ann::ConvNet::forward(torch::Tensor x)
{
	assert(x.dim() < 3);
	if(x.dim() == 2)
		x = x.reshape({x.size(0), 1, x.size(1)});
	else
		x = x.reshape({x.size(0), 1});

	return model->forward(x);
}

std::shared_ptr<torch::nn::Module> ann::ConvNet::operator[](size_t index)
{
	return model->operator[](index);
}

void ann::ConvNet::getConfiguration(Json::Value& node)
{
	node["type"] = typeid(*this).name();
	node["downsampleSteps"] = downsampleSteps;
	node["extraSteps"] = extraSteps;
	ann::Net::getConfiguration(node);
}
