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

	torch::Tensor predictedSpectraCmplx = runScriptModel(*model, modelScript, output, omegas).t();

	/*std::cout<<"targetSpectra:\n"<<targetSpectra<<'\n';
	std::cout<<"predictedSpectraCmplx:\n"<<predictedSpectraCmplx<<'\n';*/

	torch::Tensor out = torch::sum(torch::abs(predictedSpectraCmplx - targetSpectra))/targetSpectra.numel();
	//torch::Tensor out = loss->forward(torch::abs(targetSpectra), torch::abs(predictedSpectraCmplx))/targetSpectra.numel();
	//torch::Tensor out = torch::abs(torch::sum(torch::sqrt(torch::pow(predictedSpectraCmplx - targetSpectra, 2)))/targetSpectra.numel());

	return out;
}

torch::Tensor EisDistanceLoss::forward(torch::Tensor output, torch::Tensor targets)
{
	assert(targets.sizes() == output.sizes());

	output = output/targetScalar;
	targets = targets/targetScalar;

	if(targets.dim() > 1 && targets.size(1) == static_cast<int64_t>(model->getParameterCount()) )
		targets = targets.t();

	torch::Tensor targetSpectraCmplx = runScriptModel(*model, modelScript, targets, omegas).t();

	return distance(output, targetSpectraCmplx);
}
