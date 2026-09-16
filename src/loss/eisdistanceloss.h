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
 *
 * forward() applies log1p to that per point error before averaging: the raw value is heavy tailed
 * and a single point where the target spectrum crosses zero would dominate the gradient. distance()
 * keeps the raw mean so it stays usable as a metric.
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
	torch::Tensor parameterLog;
	double weightFloor;

	/**
	 * Bring the parameters of a prediction and the spectra of a target into the shape the
	 * comparison needs: one row per frequency, one column per element of the batch.
	 */
	std::pair<torch::Tensor, torch::Tensor> prepareSpectra(torch::Tensor output, torch::Tensor targetSpectra) const;
	/**
	 * The squared relative deviation of every single frequency point of a predicted spectrum,
	 * weighted by the impedance of the target at that point.
	 */
	torch::Tensor relativeErrorPerPoint(const torch::Tensor& predicted, const torch::Tensor& target) const;

	public:
		EisDistanceLoss(std::string modelString, torch::Tensor omegas, torch::Tensor parameterScalar = torch::Tensor(),
				torch::Tensor parameterBias = torch::Tensor(), torch::Tensor parameterLog = torch::Tensor(),
				double weightFloor = DEFAULT_WEIGHT_FLOOR);
		EisDistanceLoss(const eis::Model& model, torch::Tensor omegas, torch::Tensor parameterScalar = torch::Tensor(),
				torch::Tensor parameterBias = torch::Tensor(), torch::Tensor parameterLog = torch::Tensor(),
				double weightFloor = DEFAULT_WEIGHT_FLOOR);

		torch::Tensor forward(torch::Tensor output, torch::Tensor targets);
		torch::Tensor distance(torch::Tensor output, torch::Tensor targetSpectra);
		inline torch::Tensor operator()(torch::Tensor output, torch::Tensor targets)
		{
			return forward(output, targets);
		}
};
