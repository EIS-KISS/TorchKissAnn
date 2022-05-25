#pragma once

#include <cstdint>
struct Net : torch::nn::Module
{
	const int64_t inputSize;
	const int64_t outputSize;

	Net(int64_t inputSizeI = 100, int64_t outputSizeI = 6);
	torch::Tensor forward(torch::Tensor x);
	// Use one of many "standard library" modules.
	torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, fc4{nullptr};
	int64_t getInputSize();
	int64_t getOutputSize();
};
