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

#include "torchph.h"
#include "loss/eisdistanceloss.h"
#include "loss/permutationmsseloss.h"

/**
 * Sum of the spectra-domain EisDistanceLoss and the parameter-domain PermutationMseLoss.
 *
 * The permutation term alone scores an isotropic error in (log) parameter space and reliably
 * learns the shape of the spectra but leaves the overall magnitude and the series resistance at
 * the prior mean. The distance term routes its gradient through the circuit, so it is the term
 * that actually moves |Z| and the low frequency intercept; the permutation term keeps the
 * parameter space accurate and resolves the equivalent parameter degeneracy. Both terms take the
 * normalized network output and target and undo the normalisation themselves.
 */
class EisCompositeLoss : public torch::nn::Module
{
	// The two terms live on different scales (the permutation mse sits around 0.5, the mean log
	// relative spectra error around 0.01-0.1). Equal weights keep the spectra gradient from being
	// drowned out by the larger permutation value without letting it take over the training.
	static constexpr double WEIGHT_EIS = 1.0;
	static constexpr double WEIGHT_PERM = 1.0;

	std::shared_ptr<EisDistanceLoss> lossEis;
	std::shared_ptr<PermutationMseLoss> lossPermutation;

	public:
		EisCompositeLoss(std::string modelString, torch::Tensor omegas, torch::Tensor parameterScalar = torch::Tensor(),
				torch::Tensor parameterBias = torch::Tensor(), torch::Tensor parameterLog = torch::Tensor());

		torch::Tensor forward(torch::Tensor output, torch::Tensor targets);
		inline torch::Tensor operator()(torch::Tensor output, torch::Tensor targets)
		{
			return forward(output, targets);
		}
};
