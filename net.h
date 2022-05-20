#pragma once

struct Net : torch::nn::Module
{
	static constexpr int64_t INPUT_SIZE = 100;
	static constexpr int64_t OUTPUT_SIZE = 7;
	Net();
	torch::Tensor forward(torch::Tensor x);
	// Use one of many "standard library" modules.
	torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, fc4{nullptr};
	int64_t getInputSize();
	int64_t getOutputSize();
};
