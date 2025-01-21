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

#include <eisgenerator/model.h>
#include <unistd.h>
#include <memory>

#include "torchph.h"

class EisDistanceLoss : public  torch::nn::Module
{
	std::shared_ptr<eis::Model> model;
	torch::Tensor omegas;
	std::shared_ptr<torch::CompilationUnit> modelScript;
	torch::nn::MSELoss loss;
	torch::Tensor targetScalar;

	public:
		EisDistanceLoss(std::string modelString, torch::Tensor omegas, torch::Tensor targetScalar = torch::Tensor());
		EisDistanceLoss(const eis::Model& model, torch::Tensor omegas, torch::Tensor targetScalar = torch::Tensor());

		torch::Tensor forward(torch::Tensor output, torch::Tensor targets);
		torch::Tensor distance(torch::Tensor output, torch::Tensor targetSpectra);
		inline torch::Tensor operator()(torch::Tensor output, torch::Tensor targets)
		{
			return forward(output, targets);
		}
};
