#pragma once
#include <torch/torch.h>

struct Net : torch::nn::Module
{
	Net();
	torch::Tensor forward(torch::Tensor x);
	// Use one of many "standard library" modules.
	torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};
