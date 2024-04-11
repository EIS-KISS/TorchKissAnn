#pragma once

#include "net.h"

namespace ann
{

class ConvNet : public Net
{
	torch::nn::Sequential model;
	size_t downsampleSteps;
	size_t extraSteps;

	void init();

public:
	ConvNet(const Json::Value& node);
	ConvNet( int64_t inputSizeI = 100, int64_t outputSizeI = 6, size_t downsampleStepsI = 4, size_t extraStepsI = 3, bool softmax = true);
	virtual torch::Tensor forward(torch::Tensor x) override;
	virtual void getConfiguration(Json::Value& node) override;
	virtual std::shared_ptr<torch::nn::Module> operator[](size_t index) override;
};

}
