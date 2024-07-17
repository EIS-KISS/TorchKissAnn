#pragma once

#include "eisdataset.h"

template <typename DataSelf>
class RegressionDataset: public EisDataset<DataSelf>
{
public:
	virtual std::pair<torch::Tensor, torch::Tensor> getTargetScalesAndBias() = 0;
};

