#pragma once
#include <kisstype/type.h>

#include "eisdataset.h"
#include "tensoroptions.h"

template <typename DataSelf>
class RegressionDataset: public EisDataset<DataSelf>
{
public:
	virtual std::pair<torch::Tensor, torch::Tensor> getTargetScalesAndBias();
	virtual const std::string targetName();
};

template <typename DataSelf>
std::pair<torch::Tensor, torch::Tensor> RegressionDataset<DataSelf>::getTargetScalesAndBias()
{
	torch::Tensor max = torch::zeros({static_cast<int>(this->outputSize())}, tensorOptCpu<fvalue>(false));
	torch::Tensor min = torch::full({static_cast<int>(this->outputSize())}, std::numeric_limits<fvalue>::max(),  tensorOptCpu<fvalue>(false));

	for(size_t i = 0; i < this->size().value(); ++i)
	{
		torch::Tensor targets = this->getImpl(i).target;
		max = torch::maximum(targets, max);
		min = torch::minimum(targets, min);
	}
	return {(max-min), min};
}

template <typename DataSelf> const std::string RegressionDataset<DataSelf>::targetName()
{
	return "Unkown";
}

