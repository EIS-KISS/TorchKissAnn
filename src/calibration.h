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

#include <memory>
#include <string>
#include <torch/torch.h>
#include <vector>

#include "net.h"

namespace calibration
{

class Accumulator
{
	public:
		Accumulator(const std::shared_ptr<ann::Net>& net);

		void add(const torch::Tensor& inputs, const torch::Tensor& targets);

		bool empty() const
		{
			return count == 0;
		}

		torch::Tensor meanOutput() const;
		torch::Tensor meanTarget() const;

	private:
		std::shared_ptr<ann::Net> network;
		torch::Tensor outputSum;
		torch::Tensor targetSum;
		double count = 0;
};

bool apply(const std::shared_ptr<ann::Net>& net, const std::vector<Accumulator>& points);

// Log the scale and bias the calibration produced, per output label.
void report(const std::shared_ptr<ann::Net>& net, const torch::Tensor& scale,
		const torch::Tensor& bias);

}
