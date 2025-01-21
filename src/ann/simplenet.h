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

#include "net.h"

namespace ann
{

class SimpleNet : public Net
{
	torch::nn::Sequential model;
	size_t downsampleSteps;
	size_t extraSteps;

	void init();

public:
	SimpleNet(const Json::Value& node);
	SimpleNet(int64_t inputSizeI = 100, int64_t outputSizeI = 6, size_t downsampleSteps = 4, size_t extraSteps = 3, bool softmax = true);
	virtual torch::Tensor forward(torch::Tensor x) override;
	virtual void getConfiguration(Json::Value& node) override;
	virtual std::shared_ptr<torch::nn::Module> operator[](size_t index) override;
};

}
