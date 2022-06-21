#pragma once

#include <cstdint>

namespace ann
{

class Net : public torch::nn::Module
{
protected:
	const int64_t inputSize;
	const int64_t outputSize;
	const bool leak = false;

public:
	Net(int64_t inputSizeI = 100, int64_t outputSizeI = 6, bool leakI = false):
	inputSize(inputSizeI), outputSize(outputSizeI), leak(leakI)
	{}

	virtual torch::Tensor forward(torch::Tensor x) = 0;

	int64_t getInputSize();
	int64_t getOutputSize();
};

class SimpleNet : public Net
{
	torch::nn::Linear fc1{nullptr};
	torch::nn::Linear fc2{nullptr};
	torch::nn::Linear fc3{nullptr};
	torch::nn::Linear fc4{nullptr};

public:
	SimpleNet(int64_t inputSizeI = 100, int64_t outputSizeI = 6, bool leakI = false);
	virtual torch::Tensor forward(torch::Tensor x) override;
};

class ConvNet : public Net
{
	torch::nn::Sequential model;

public:
	ConvNet(int64_t inputSizeI = 100, int64_t outputSizeI = 6, bool leakI = false);
	virtual torch::Tensor forward(torch::Tensor x) override;
};

}
