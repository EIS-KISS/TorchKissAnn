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
#include <tensoroperators.h>

#include <vector>

torch::Tensor eisToComplexTensor(const std::vector<eis::DataPoint>& data, torch::Tensor* freqs = nullptr);
torch::Tensor eisToTorch(const std::vector<eis::DataPoint>& data, torch::Tensor* freqs = nullptr);
torch::Tensor eisToTorchExtra(const std::vector<eis::DataPoint>& data, std::vector<fvalue> extraInputs = {});
std::vector<eis::DataPoint> torchToEis(const torch::Tensor& input);
torch::Tensor fvalueVectorToTensor(std::vector<fvalue>& vect);

torch::Tensor rangeToTensor(const eis::Range& range);
eis::Range tensorToRange(const torch::Tensor& tensor);

template<typename fv>
std::pair<std::valarray<fv>, std::valarray<fv>> torchToValarray(torch::Tensor tensor)
{
	assert(tensor.numel() % 2 == 0);
	assert(checkTorchType<fv>(tensor));
	torch::Tensor work = tensor.reshape({tensor.numel()});

	std::pair<std::valarray<fv>, std::valarray<fv>> out(std::valarray<fv>(0.0, tensor.numel()/2), std::valarray<fv>(0.0, tensor.numel()/2));
	auto accessor = work.accessor<fv, 1>();
	for(int64_t i = 0; i < tensor.numel()/2; ++i)
		out.first[i] = accessor[i];
	for(int64_t i = tensor.numel()/2; i < tensor.numel(); ++i)
		out.second[i-tensor.numel()/2] = accessor[i];
	return out;
}
