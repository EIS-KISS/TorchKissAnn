#pragma once
#include "log.h"

class SeqPrint : public torch::nn::Module
{
public:
	SeqPrint() = default;

	torch::Tensor forward(torch::Tensor x)
	{
		Log(Log::DEBUG)<<x.sizes();
		std::cout<<std::flush;
		return x;
	}
};
