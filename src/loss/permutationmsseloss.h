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

#include <cstddef>
#include <cstdint>
#include <eisgenerator/model.h>
#include <map>
#include <string>
#include <vector>

#include "torchph.h"

class PermutationMseLoss : public torch::nn::Module
{
	static constexpr double DEFAULT_TOLERANCE = 1e-3;
	static constexpr size_t DEFAULT_LIMIT = 4096;

	std::string modelString;
	int64_t parameterCount = 0;
	eis::Range omegaRange;
	torch::Tensor parameterScalar;
	torch::Tensor parameterBias;
	torch::Tensor parameterLog;
	double tolerance;
	size_t limit;
	std::map<std::vector<double>, std::vector<std::vector<double>>> cache;

	const std::vector<std::vector<double>>& candidatesFor(const std::vector<double>& parameters);
	// {candidateCount, batch, parameterCount} difference between the predictions and every
	// candidate permutation, in the space the network works in
	torch::Tensor prepareDifference(torch::Tensor output, torch::Tensor targets);

	public:
		PermutationMseLoss(std::string modelString, torch::Tensor omegas, torch::Tensor parameterScalar = torch::Tensor(),
				torch::Tensor parameterBias = torch::Tensor(), torch::Tensor parameterLog = torch::Tensor(),
				double tolerance = DEFAULT_TOLERANCE, size_t limit = DEFAULT_LIMIT);

		torch::Tensor forward(torch::Tensor output, torch::Tensor targets);
		// Like forward, but instead of collapsing to a single scalar, returns the mse of the
		// best permutation per parameter, in the order of the model's parameters
		torch::Tensor forwardPerParameter(torch::Tensor output, torch::Tensor targets);
		inline torch::Tensor operator()(torch::Tensor output, torch::Tensor targets)
		{
			return forward(output, targets);
		}
};
