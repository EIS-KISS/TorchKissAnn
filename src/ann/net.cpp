#include "net.h"
#include "../utils/seqprint.h"
#include "log.h"

using namespace ann;

int64_t ann::Net::getInputSize()
{
	return inputSize;
}

int64_t ann::Net::getOutputSize()
{
	return outputSize;
}

static std::pair<size_t, size_t> layerWidthFunction(size_t layer, size_t leyers, size_t input, size_t output)
{
	size_t layerIn = (input*(leyers-layer)/leyers)+output*(layer)/leyers;
	size_t layerOut = (input*(leyers-layer-1)/leyers)+output*(layer+1)/leyers;
	return std::pair<size_t, size_t>(layerIn, layerOut);
}

ann::SimpleNet::SimpleNet(int64_t inputSizeI, int64_t outputSizeI, size_t downsampleSteps, size_t extraSteps):
Net(inputSizeI, outputSizeI)
{
	register_module("SimpleModel", model);

	std::pair<size_t, size_t> sizes = layerWidthFunction(0, downsampleSteps, inputSize, outputSize);
	torch::nn::Linear linear(sizes.first, sizes.first);
	torch::nn::init::xavier_uniform_(linear->weight);
	model->push_back(linear);
	model->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)));
	for(size_t i = 0; i < downsampleSteps-1; ++i)
	{
		sizes = layerWidthFunction(i, downsampleSteps, inputSize, outputSize);
		torch::nn::Linear linear(sizes.first, sizes.second);
		torch::nn::init::xavier_uniform_(linear->weight);
		model->push_back(linear);
		model->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1).inplace(true)));
		/*if(i % 2 == 0)
			model->push_back(torch::nn::Dropout(torch::nn::DropoutOptions().p(0.1)));*/
		torch::nn::BatchNorm1d bn(torch::nn::BatchNorm1dOptions(sizes.second).eps(0.5).momentum(0.1).affine(false));
		model->push_back(bn);
	}
	sizes = layerWidthFunction(downsampleSteps-1, downsampleSteps, inputSize, outputSize);
	model->push_back(torch::nn::Linear(sizes.first, sizes.second));
	model->push_back(torch::nn::LogSoftmax(torch::nn::LogSoftmaxOptions(1)));
}

torch::Tensor ann::SimpleNet::forward(torch::Tensor x)
{
	return model->forward(x);
}

ann::ConvNet::ConvNet(int64_t inputSizeI, int64_t outputSizeI, size_t downsampleSteps, size_t extraSteps):
Net(inputSizeI, outputSizeI)
{
	int64_t size = inputSize;
	const int conv1Kernel = inputSize/5;
	const int maxPool1Devisor = 2;
	const int conv2Kernel = inputSize/10;
	const int maxPool2Devisor = 2;

	register_module("ConvModel", model);

	//conv part
	//model->push_back(SeqPrint());
	model->push_back(torch::nn::Conv1d(torch::nn::Conv1dOptions(1, 3, conv1Kernel).stride(1).bias(false)));
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
