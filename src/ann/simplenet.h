#pragma once

#include "net.h"

namespace ann
{

class SimpleNet : public Net
{
	torch::nn::Sequential model;
	size_t downsampleSteps;
	size_t extraSteps;

	void init();

public:
	SimpleNet(const Json::Value& node);
	SimpleNet(int64_t inputSizeI = 100, int64_t outputSizeI = 6, size_t downsampleSteps = 4, size_t extraSteps = 3, bool softmax = true);
	virtual torch::Tensor forward(torch::Tensor x) override;
	virtual void getConfiguration(Json::Value& node) override;
	virtual std::shared_ptr<torch::nn::Module> operator[](size_t index) override;
};

}
