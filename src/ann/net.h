#pragma once

#include <cstdint>

namespace ann
{

class Net : public torch::nn::Module
{
protected:
	const int64_t inputSize;
	const int64_t outputSize;

public:
	Net(int64_t inputSizeI, int64_t outputSizeI):
	inputSize(inputSizeI), outputSize(outputSizeI)
	{}

	virtual torch::Tensor forward(torch::Tensor x) = 0;

	int64_t getInputSize();
	int64_t getOutputSize();
};

class SimpleNet : public Net
{
	torch::nn::Sequential model;

public:
	SimpleNet(int64_t inputSizeI = 100, int64_t outputSizeI = 6, size_t downsampleSteps = 4, size_t extraSteps = 3);
	virtual torch::Tensor forward(torch::Tensor x) override;
};

class ConvNet : public Net
{
	torch::nn::Sequential model;

public:
	ConvNet(int64_t inputSizeI = 100, int64_t outputSizeI = 6, size_t downsampleSteps = 4, size_t extraSteps = 3);
	virtual torch::Tensor forward(torch::Tensor x) override;
};

}
