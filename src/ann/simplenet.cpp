#include "simplenet.h"
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

ann::SimpleNet::SimpleNet(const Json::Value& node):
Net(node)
{
	downsampleSteps = node["downsampleSteps"].asUInt64();
	extraSteps = node["extraSteps"].asUInt64();
	init();
}

ann::SimpleNet::SimpleNet(int64_t inputSizeI, int64_t outputSizeI, size_t downsampleStepsI, size_t extraStepsI, bool softmaxI):
Net(inputSizeI, outputSizeI, softmaxI), downsampleSteps(downsampleStepsI), extraSteps(extraStepsI)
{
	init();
}

void ann::SimpleNet::init()
{
	register_module("SimpleModel", model);

	std::pair<size_t, size_t> sizes = layerWidthFunction(0, downsampleSteps, inputSize, outputSize);
	torch::nn::Linear linear(sizes.first, sizes.first);
	model->push_back(linear);
	model->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)));
	for(size_t i = 0; i < downsampleSteps-1; ++i)
	{
		sizes = layerWidthFunction(i, downsampleSteps, inputSize, outputSize);
		torch::nn::Linear linear(sizes.first, sizes.second);
		model->push_back(linear);
		model->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1).inplace(true)));
		/*if(i % 2 == 0)
			model->push_back(torch::nn::Dropout(torch::nn::DropoutOptions().p(0.2)));*/
		torch::nn::BatchNorm1d bn(torch::nn::BatchNorm1dOptions(sizes.second).eps(1e-3).momentum(0.1).affine(true));
		model->push_back(bn);
		if(i == 0)
		{
			for(size_t j = 0; j < extraSteps; ++j)
			{
				torch::nn::Linear linear(sizes.second, sizes.second);
				model->push_back(linear);
				model->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)));
			}
		}
	}
	sizes = layerWidthFunction(downsampleSteps-1, downsampleSteps, inputSize, outputSize);
	linear = torch::nn::Linear(sizes.first, sizes.second);
	model->push_back(linear);
	if(softmax)
		model->push_back(torch::nn::LogSoftmax(torch::nn::LogSoftmaxOptions(1)));
}

std::shared_ptr<torch::nn::Module> ann::SimpleNet::operator[](size_t index)
{
	return model->operator[](index);
}

torch::Tensor ann::SimpleNet::forward(torch::Tensor x)
{
	return model->forward(x);
}

void ann::SimpleNet::getConfiguration(Json::Value& node)
{
	node["type"] = typeid(*this).name();
	node["downsampleSteps"] = downsampleSteps;
	node["extraSteps"] = extraSteps;
	ann::Net::getConfiguration(node);
}
