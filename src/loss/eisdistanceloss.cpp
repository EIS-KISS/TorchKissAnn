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

#include "modelscript.h"
#include "tensoroperators.h"
#include "log.h"

EisDistanceLoss::EisDistanceLoss(std::string modelString, torch::Tensor omegasIn, torch::Tensor parameterScalarIn,
		torch::Tensor parameterBiasIn, torch::Tensor parameterLogIn, double weightFloorIn):
omegas(omegasIn.detach().reshape({omegasIn.numel(), -1})),
parameterScalar(parameterScalarIn),
parameterBias(parameterBiasIn),
parameterLog(parameterLogIn),
weightFloor(weightFloorIn)
{
	model = std::shared_ptr<eis::Model>(new eis::Model(modelString));
	modelScript = compileModel(*model);
}


EisDistanceLoss::EisDistanceLoss(const eis::Model& modelIn, torch::Tensor omegasIn, torch::Tensor parameterScalarIn,
		torch::Tensor parameterBiasIn, torch::Tensor parameterLogIn, double weightFloorIn):
omegas(omegasIn.detach().reshape({omegasIn.numel(), -1})),
parameterScalar(parameterScalarIn),
parameterBias(parameterBiasIn),
parameterLog(parameterLogIn),
weightFloor(weightFloorIn)
{
	model = std::shared_ptr<eis::Model>(new eis::Model(modelIn));
	modelScript = compileModel(*model);
}

std::pair<torch::Tensor, torch::Tensor> EisDistanceLoss::prepareSpectra(torch::Tensor output, torch::Tensor targetSpectra) const
{
	assert(model);
	assert(omegas.numel() > 0);
	assert(modelScript);

	output = output.to(torch::kCPU);
	targetSpectra = targetSpectra.to(torch::kCPU);

	if(output.dim() > 1 && output.size(1) == static_cast<int64_t>(model->getParameterCount()))
		output = output.t();

	torch::Tensor predictedSpectraCmplx = runScriptModel(*model, modelScript, output, omegas);

	torch::Tensor predicted = predictedSpectraCmplx.reshape({omegas.numel(), -1});
	torch::Tensor target = targetSpectra.detach().reshape({omegas.numel(), -1});

	return {predicted, target};
}

torch::Tensor EisDistanceLoss::relativeErrorPerPoint(const torch::Tensor& predicted, const torch::Tensor& target) const
{
	torch::Tensor diffReal = torch::real(predicted) - torch::real(target);
	torch::Tensor diffImag = torch::imag(predicted) - torch::imag(target);

	torch::Tensor targetMagnitude = torch::abs(target);
	torch::Tensor floor = torch::mean(targetMagnitude, 0, true)*weightFloor;

	return (diffReal*diffReal + diffImag*diffImag)/(targetMagnitude*targetMagnitude + floor*floor);
}

torch::Tensor EisDistanceLoss::distance(torch::Tensor output, torch::Tensor targetSpectra)
{
	const torch::Device computeDevice = output.device();

	std::pair<torch::Tensor, torch::Tensor> spectra = prepareSpectra(output, targetSpectra);

	return torch::mean(relativeErrorPerPoint(spectra.first, spectra.second)).to(computeDevice);
}

torch::Tensor EisDistanceLoss::forward(torch::Tensor output, torch::Tensor targets)
{
	assert(targets.sizes() == output.sizes());

	const torch::Device computeDevice = output.device();

	if(parameterScalar.numel() != 0)
	{
		// the model script is only defined for physical parameters, undo the dataset normalisation
		output = affineToPhysical(output, parameterScalar, parameterBias, parameterLog);
		targets = affineToPhysical(targets, parameterScalar, parameterBias, parameterLog);
	}

	if(targets.dim() > 1 && targets.size(1) == static_cast<int64_t>(model->getParameterCount()) )
		targets = targets.t();
	targets = targets.to(torch::kCPU);
	torch::Tensor targetSpectraCmplx = runScriptModel(*model, modelScript, targets, omegas);

	std::pair<torch::Tensor, torch::Tensor> spectra = prepareSpectra(output, targetSpectraCmplx);

	return torch::mean(torch::log1p(relativeErrorPerPoint(spectra.first, spectra.second))).to(computeDevice);
}
