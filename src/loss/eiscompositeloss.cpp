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

#include "eiscompositeloss.h"

#include "log.h"

EisCompositeLoss::EisCompositeLoss(std::string modelString, torch::Tensor omegas, torch::Tensor parameterScalar,
		torch::Tensor parameterBias, torch::Tensor parameterLog):
lossEis(std::make_shared<EisDistanceLoss>(modelString, omegas, parameterScalar, parameterBias, parameterLog)),
lossPermutation(std::make_shared<PermutationMseLoss>(modelString, omegas, parameterScalar, parameterBias, parameterLog))
{
	Log(Log::INFO)<<"Composite loss: "<<WEIGHT_EIS<<" * EisDistanceLoss + "<<WEIGHT_PERM<<" * PermutationMseLoss on "<<modelString;
}

torch::Tensor EisCompositeLoss::forward(torch::Tensor output, torch::Tensor targets)
{
	torch::Tensor eis = lossEis->forward(output, targets);
	torch::Tensor permutation = lossPermutation->forward(output, targets);
	return WEIGHT_EIS*eis + WEIGHT_PERM*permutation;
}
