/* * TorchKissAnn - A collection of tools to train various types of Machine learning
 * algorithms on various types of EIS data
 * Copyright (C) 2025 Carl Klemm <carl@uvos.xyz>
 *
 * This file is part of TorchKissAnn.
 *
 * TorchKissAnn is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * TorchKissAnn is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with TorchKissAnn.  If not, see <http://www.gnu.org/licenses/>.
 */

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

