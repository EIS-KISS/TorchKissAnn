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
#include <cstdint>
#include <kisstype/type.h>

#include "eisdataset.h"
#include "tensoroperators.h"
#include "tensoroptions.h"

struct RegressionNormalization
{
	torch::Tensor inputScalars; // Multiplier and offset applied to the (log compressed) input so the observed range of every feature maps onto [-1, 1].
	torch::Tensor inputBiases;
	int64_t inputLogCount = 0;
	double inputLogFloor = ann::Net::DEFAULT_INPUT_LOG_FLOOR;
	torch::Tensor targetScalars;
	torch::Tensor targetBiases;
	torch::Tensor targetLogMask;
};

template <typename DataSelf>
class RegressionDataset: public EisDataset<DataSelf>
{
public:
	virtual std::pair<torch::Tensor, torch::Tensor> getTargetScalesAndBias();
	virtual RegressionNormalization getNormalization();
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

template <typename DataSelf>
RegressionNormalization RegressionDataset<DataSelf>::getNormalization()
{
	const int64_t inSize = static_cast<int64_t>(this->inputSize());
	const int64_t outSize = static_cast<int64_t>(this->outputSize());

	torch::Tensor inMax = torch::full({inSize}, std::numeric_limits<fvalue>::lowest(), tensorOptCpu<fvalue>(false));
	torch::Tensor inMin = torch::full({inSize}, std::numeric_limits<fvalue>::max(), tensorOptCpu<fvalue>(false));
	torch::Tensor outMax = torch::full({outSize}, std::numeric_limits<fvalue>::lowest(), tensorOptCpu<fvalue>(false));
	torch::Tensor outMin = torch::full({outSize}, std::numeric_limits<fvalue>::max(), tensorOptCpu<fvalue>(false));

	for(size_t i = 0; i < this->size().value(); ++i)
	{
		torch::data::Example<torch::Tensor, torch::Tensor> example = this->getImpl(i);
		torch::Tensor data = example.data.reshape({-1});
		torch::Tensor targets = example.target.reshape({-1});
		inMax = torch::maximum(data, inMax);
		inMin = torch::minimum(data, inMin);
		outMax = torch::maximum(targets, outMax);
		outMin = torch::minimum(targets, outMin);
	}

	RegressionNormalization norm;

	int64_t extraCount = 0;
	for(const std::pair<std::string, int64_t>& extra : this->extraInputs())
		extraCount += extra.second;
	norm.inputLogCount = std::max<int64_t>(0, std::min(inSize, inSize-extraCount));

	torch::Tensor inMaxT = signedLog10Leading(inMax, norm.inputLogCount, norm.inputLogFloor);
	torch::Tensor inMinT = signedLog10Leading(inMin, norm.inputLogCount, norm.inputLogFloor);

	torch::Tensor inHalf = 0.5*(inMaxT-inMinT);
	torch::Tensor inMid = 0.5*(inMaxT+inMinT);
	norm.inputScalars = torch::where(inHalf > 0, 1.0/torch::clamp(inHalf, 1e-30, 1e30), torch::zeros_like(inHalf));
	norm.inputBiases = -inMid*norm.inputScalars;

	torch::Tensor logMin = torch::log10(torch::clamp(outMin, 1e-30, 1e30));
	torch::Tensor logMax = torch::log10(torch::clamp(outMax, 1e-30, 1e30));
	torch::Tensor useLog = ((outMin > 0) & ((logMax-logMin) >= 1.0)).to(torch::kFloat32);
	torch::Tensor workMin = torch::where(useLog > 0, logMin, outMin);
	torch::Tensor workMax = torch::where(useLog > 0, logMax, outMax);
	torch::Tensor outHalf = 0.5*(workMax-workMin);
	torch::Tensor outMid = 0.5*(workMax+workMin);
	norm.targetScalars = torch::where(outHalf > 0, outHalf, torch::ones_like(outHalf));
	norm.targetBiases = outMid;
	norm.targetLogMask = useLog;

	return norm;
}

template <typename DataSelf> const std::string RegressionDataset<DataSelf>::targetName()
{
	return "Unkown";
}

