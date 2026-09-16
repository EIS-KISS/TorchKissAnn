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
// the frequencies are constants, but the dataset hands them out as tensors which require grad.
// They are inputs to complex operations, so autograd would try to accumulate a complex gradient
// into a real tensor and abort in backward()
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

	if(output.dim() > 1 && output.size(1) == static_cast<int64_t>(model->getParameterCount()))
		output = output.t();

	torch::Tensor predictedSpectraCmplx = runScriptModel(*model, modelScript, output, omegas);

	// the script output is omega major, so this puts every frequency in one row and every
	// element of the batch in one column
	torch::Tensor predicted = predictedSpectraCmplx.reshape({omegas.numel(), -1});
	// the target spectra is data, the loss is only differentiated with respect to the prediction
	torch::Tensor target = targetSpectra.detach().reshape({omegas.numel(), -1});

	return {predicted, target};
}

torch::Tensor EisDistanceLoss::relativeErrorPerPoint(const torch::Tensor& predicted, const torch::Tensor& target) const
{
	torch::Tensor diffReal = torch::real(predicted) - torch::real(target);
	torch::Tensor diffImag = torch::imag(predicted) - torch::imag(target);

	// weight every point by the impedance of the target at that point, otherwise a single
	// frequency at which the impedance is large dominates the loss and the gradient. The floor
	// keeps points at which the target spectrum approaches zero finite.
	torch::Tensor targetMagnitude = torch::abs(target);
	torch::Tensor floor = torch::mean(targetMagnitude, 0, true)*weightFloor;

	return (diffReal*diffReal + diffImag*diffImag)/(targetMagnitude*targetMagnitude + floor*floor);
}

torch::Tensor EisDistanceLoss::distance(torch::Tensor output, torch::Tensor targetSpectra)
{
	std::pair<torch::Tensor, torch::Tensor> spectra = prepareSpectra(output, targetSpectra);

	// mean instead of sum, so the loss does not scale with the batch size and is comparable
	// between runs and between samples of different absolute impedance
	return torch::mean(relativeErrorPerPoint(spectra.first, spectra.second));
}

torch::Tensor EisDistanceLoss::forward(torch::Tensor output, torch::Tensor targets)
{
	assert(targets.sizes() == output.sizes());

	if(parameterScalar.numel() != 0)
	{
		// the model script is only defined for physical parameters, undo the dataset normalisation
		output = affineToPhysical(output, parameterScalar, parameterBias, parameterLog);
		targets = affineToPhysical(targets, parameterScalar, parameterBias, parameterLog);
	}

	if(targets.dim() > 1 && targets.size(1) == static_cast<int64_t>(model->getParameterCount()) )
		targets = targets.t();

	torch::Tensor targetSpectraCmplx = runScriptModel(*model, modelScript, targets, omegas);

	std::pair<torch::Tensor, torch::Tensor> spectra = prepareSpectra(output, targetSpectraCmplx);

	// the raw relative error is heavy tailed: a point at which the target spectrum is close to
	// zero can contribute more than every other point together, so the optimiser spends its
	// gradient there. log1p keeps the ordering and the small error behaviour but bounds what a
	// single point can add
	return torch::mean(torch::log1p(relativeErrorPerPoint(spectra.first, spectra.second)));
}
