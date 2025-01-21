//
// TorchKissAnn - A collection of tools to train various types of Machine learning
// algorithms on various types of EIS data
// Copyright (C) 2025 Carl Klemm <carl@uvos.xyz>
//
// This file is part of TorchKissAnn.
//
// TorchKissAnn is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// TorchKissAnn is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with TorchKissAnn.  If not, see <http://www.gnu.org/licenses/>.
//

#include "r2score.h"

torch::Tensor r2score(const torch::Tensor& prediction, const torch::Tensor& truth)
{
	torch::Tensor means = truth.mean(0).repeat({truth.size(0), 1});
	torch::Tensor suqareSum = torch::sum(torch::pow(truth-prediction, 2), 0);
	torch::Tensor squareSumMean = torch::sum(torch::pow(truth-means, 2), 0);
	return 1-suqareSum/squareSumMean;
}
