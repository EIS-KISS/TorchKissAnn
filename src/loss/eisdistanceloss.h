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

/**
 * Loss that compares two sets of circuit parameters by the spectra they produce.
 *
 * The distance is the mean squared relative deviation between the simulated spectra, i.e. every
 * frequency point is weighted by the impedance of the target at that point. This keeps points
 * that are many decades apart comparable and makes the value independent of the batch size and of
 * the absolute impedance of a sample.
 */
class EisDistanceLoss : public  torch::nn::Module
{
	/**
	 * Floor of the weighting, relative to the mean impedance of a sample. It keeps the weighting
	 * of points at which the target spectrum approaches zero finite.
	 */
	static constexpr double DEFAULT_WEIGHT_FLOOR = 1e-3;

	std::shared_ptr<eis::Model> model;
	torch::Tensor omegas;
	std::shared_ptr<torch::CompilationUnit> modelScript;
	torch::Tensor parameterScalar;
	torch::Tensor parameterBias;
	double weightFloor;

	public:
		EisDistanceLoss(std::string modelString, torch::Tensor omegas, torch::Tensor parameterScalar = torch::Tensor(),
				torch::Tensor parameterBias = torch::Tensor(), double weightFloor = DEFAULT_WEIGHT_FLOOR);
		EisDistanceLoss(const eis::Model& model, torch::Tensor omegas, torch::Tensor parameterScalar = torch::Tensor(),
				torch::Tensor parameterBias = torch::Tensor(), double weightFloor = DEFAULT_WEIGHT_FLOOR);

		torch::Tensor forward(torch::Tensor output, torch::Tensor targets);
		torch::Tensor distance(torch::Tensor output, torch::Tensor targetSpectra);
		inline torch::Tensor operator()(torch::Tensor output, torch::Tensor targets)
		{
			return forward(output, targets);
		}
};
