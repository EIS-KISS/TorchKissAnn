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

#include "eisdistanceloss.h"

#include <cassert>
#include <cstdint>
#include <torch/linalg.h>
#include <torch/nn/options/loss.h>

#include "modelscript.h"
#include "tensoroperators.h"
#include "log.h"

EisDistanceLoss::EisDistanceLoss(std::string modelString, torch::Tensor omegasIn, torch::Tensor targetScalarIn):
omegas(omegasIn.reshape({omegasIn.numel(), -1})),
loss(torch::nn::MSELossOptions().reduction(torch::kMean)),
targetScalar(targetScalarIn)
{
	model = std::shared_ptr<eis::Model>(new eis::Model(modelString));
	modelScript = compileModel(*model);
}


EisDistanceLoss::EisDistanceLoss(const eis::Model& modelIn, torch::Tensor omegasIn, torch::Tensor targetScalarIn):
omegas(omegasIn.reshape({omegasIn.numel(), -1})),
loss(torch::nn::MSELossOptions().reduction(torch::kMean)),
targetScalar(targetScalarIn)
{
	model = std::shared_ptr<eis::Model>(new eis::Model(modelIn));
	modelScript = compileModel(*model);
}

torch::Tensor EisDistanceLoss::distance(torch::Tensor output, torch::Tensor targetSpectra)
{
	assert(model);
	assert(omegas.numel() > 0);
	assert(modelScript);

	if(output.dim() > 1 && output.size(1) == static_cast<int64_t>(model->getParameterCount()))
		output = output.t();

	torch::Tensor predictedSpectraCmplx = runScriptModel(*model, modelScript, output, omegas);

	/*std::cout<<"targetSpectra:\n"<<torch::view_as_real(targetSpectra)<<'\n';
	std::cout<<"predictedSpectraCmplx:\n"<<torch::view_as_real(predictedSpectraCmplx)<<'\n';*/

	torch::Tensor out = /*torch::sum(torch::abs(torch::real(predictedSpectraCmplx) - torch::real(targetSpectra)))+*/
		torch::sum(torch::abs(torch::imag(predictedSpectraCmplx) - torch::imag(targetSpectra)));
/*	torch::Tensor targetSpectraExpanded = targetSpectra.expand({targetSpectra.size(0), targetSpectra.size(0)}).t();
	torch::Tensor diff = torch::abs(targetSpectraExpanded - predictedSpectraCmplx);
	std::cout<<"targetSpectraExpanded:\n"<<targetSpectraExpanded<<'\n';
	std::cout<<"Diff:\n"<<diff<<'\n';
	torch::Tensor sumA = torch::sum(std::get<0>(torch::min(diff, 1)));
	torch::Tensor sumB = torch::sum(std::get<0>(torch::min(diff, 0)));*/

	return out;//+torch::sum(torch::abs((output-torch::abs(output))))+0.001;
}

torch::Tensor EisDistanceLoss::forward(torch::Tensor output, torch::Tensor targets)
{
	assert(targets.sizes() == output.sizes());

	if(targetScalar.numel() != 0)
	{
		output = output/targetScalar;
		targets = targets/targetScalar;
	}

	if(targets.dim() > 1 && targets.size(1) == static_cast<int64_t>(model->getParameterCount()) )
		targets = targets.t();

	torch::Tensor targetSpectraCmplx = runScriptModel(*model, modelScript, targets, omegas);

	return distance(output, targetSpectraCmplx);
}
