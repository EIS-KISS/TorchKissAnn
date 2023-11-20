#include "fit.h"

#include <cstdint>
#include <eisgenerator/model.h>
#include <thread>
#include <torch/optim.h>
#include <torch/optim/adamw.h>
#include <torch/optim/lbfgs.h>

#include "modelscript.h"
#include "log.h"
#include "tensoroptions.h"
#include "loss/eisdistanceloss.h"
#include "utils/tensoroperators.h"
#include "globals.h"

std::pair<torch::Tensor, torch::Tensor> getTargetScalesAndBiases(eis::Model& model)
{
	std::vector<eis::Range> defaultRanges = model.getDefaultParameters();
	torch::Tensor scales = torch::empty({static_cast<int64_t>(defaultRanges.size())}, tensorOptCpu<double>(false));
	torch::Tensor bias = torch::empty({static_cast<int64_t>(defaultRanges.size())}, tensorOptCpu<double>(false));

	for(size_t i = 0; i < defaultRanges.size(); ++i)
	{
		scales[i] = defaultRanges[i].end-defaultRanges[i].start;
		assert(defaultRanges[i].end-defaultRanges[i].start != 0);
		bias[i] = defaultRanges[i].start;
	}
	return {scales, bias};
}

static torch::Tensor guesParameters(eis::Model& model)
{
	torch::Tensor out = torch::empty({static_cast<int64_t>(model.getParameterCount())}, tensorOptCpu<fvalue>(false));

	std::vector<eis::Range> ranges = model.getDefaultParameters();

	for(int64_t i = 0; i < out.size(0); ++i)
		out[i] = ranges[i].center();
	return out;
}

std::pair<torch::Tensor, torch::Tensor> eisFit(torch::Tensor spectra, torch::Tensor omegas, const std::string& modelString, torch::Tensor startingParams)
{
	eis::Model model(modelString);

	if(startingParams.numel() != 0 && startingParams.numel() != static_cast<int64_t>(model.getParameterCount()))
		Log(Log::WARN)<<__func__<<" starting parameters given are not of the right size for the model given";

	if(startingParams.numel() != static_cast<int64_t>(model.getParameterCount()))
		startingParams = guesParameters(model);

	startingParams[0] = 200;
	startingParams[1] = 10e-5;
	omegas.requires_grad_(false);
	EisDistanceLoss lossFn(model, omegas);

	torch::Tensor a = torch::zeros({2}, tensorOptCpu<double>(false));
	a[0] = 100;
	a[1] = 3e-6;

	std::pair<torch::Tensor, torch::Tensor> scaleAndBias = getTargetScalesAndBiases(model);

	std::cout<<"scale:\n"<<scaleAndBias.first<<'\n';
	std::cout<<"bias:\n"<<scaleAndBias.second<<'\n';

	startingParams = (startingParams-scaleAndBias.second)/scaleAndBias.first;
	startingParams.requires_grad_(true);
	startingParams.to(torch::kFloat64);

	std::vector<torch::Tensor> params;
	params.push_back(startingParams);
	torch::optim::LBFGS optimizer(params, torch::optim::LBFGSOptions(0.001).line_search_fn("strong_wolfe").history_size(1000).max_iter(1000));

	for(size_t i = 0; i < 200000; ++i)
	{
		torch::Tensor loss = lossFn.forward(params[0]*scaleAndBias.first+scaleAndBias.second, a);
		std::cout<<"loss:\n"<<loss<<'\n';
		std::cout<<"params[0]:\n"<<params[0]<<'\n';
		std::cout<<"params:\n"<<params[0]*scaleAndBias.first+scaleAndBias.second<<'\n';
		optimizer.zero_grad();
		loss.backward();
		std::cout<<"grad:\n"<<params[0].grad()<<'\n';
		optimizer.step([&lossFn, &a, parameters=params[0], &scaleAndBias]()->torch::Tensor{return lossFn.forward(parameters*scaleAndBias.first+scaleAndBias.second, a);});
		if(loss.item().toDouble() < 0.0001)
			break;
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
	}

	Log(Log::DEBUG)<<__func__<<" optimized to:\n"<<params[0]*scaleAndBias.first+scaleAndBias.second;
	torch::Tensor loss = lossFn.distance(params[0]*scaleAndBias.first+scaleAndBias.second, spectra);

	return {params[0]*scaleAndBias.first+scaleAndBias.second, loss};
}
