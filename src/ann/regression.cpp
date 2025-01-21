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

#include "regression.h"
#include "ann/convnet.h"
#include "ann/simplenet.h"

#include <cstdint>
#include <kisstype/type.h>
#include <memory>
#include <torch/optim/adamw.h>


using namespace ann::regression;

torch::Tensor ann::regression::use(torch::Tensor input, std::shared_ptr<Net> net)
{
	torch::NoGradGuard noGrad;
	net->eval();
	torch::Tensor output = net->forward(input);
	torch::Tensor outputScalars = net->getOutputScalars();
	torch::Tensor outputBias    = net->getOutputBiases();
	output = (output*outputScalars+outputBias);
	return output;
}

std::pair<torch::Tensor, torch::Tensor>  ann::regression::getTargetScalesAndBiases(const std::string& modelString)
{
	try
	{
		eis::Model model(modelString);
		if(model.isReady())
		{
			std::vector<eis::Range> defaultRanges = model.getDefaultParameters();
			torch::Tensor scales = torch::empty({static_cast<int64_t>(defaultRanges.size())}, tensorOptCpu<fvalue>(false));
			torch::Tensor bias = torch::empty({static_cast<int64_t>(defaultRanges.size())}, tensorOptCpu<fvalue>(false));

			for(size_t i = 0; i < defaultRanges.size(); ++i)
			{
				scales[i] = defaultRanges[i].end-defaultRanges[i].start;
				bias[i] = defaultRanges[i].start;
			}
			return {scales, bias};
		}
	}
	catch(const eis::parse_errror& err)
	{
		return {torch::Tensor(), torch::Tensor()};
	}

	return {torch::Tensor(), torch::Tensor()};
}

struct Configuration
{
	int extraSteps;
	int downsampleSteps;
	bool conv;

	void print(Log::Level level)
	{
		Log(level)<<"Configuration:\n\t"<<"extraSteps = "<<extraSteps<<"\n\tdownsampleSteps = "
			<<downsampleSteps<<"\n\tconv = "<<(conv ? "true" : "false");
	}
};

/*static double tryConfiguration(const Configuration config)
{
	std::shared_ptr<ann::Net> net;

	ParameterRegressionDatasetGenerator dataset("rp", 25, 0, false);

	if(config.conv)
	{
		net = std::shared_ptr<ann::Net>(new ann::ConvNet(dataset.get(0).data.numel(),
		                                dataset.outputSize(), config.downsampleSteps, config.extraSteps, false));
	}
	else
	{
		net = std::shared_ptr<ann::Net>(new ann::SimpleNet(dataset.get(0).data.numel(),
		                                dataset.outputSize(), config.downsampleSteps, config.extraSteps, false));
	}

	double out;

	train<ParameterRegressionDatasetGenerator, ParameterRegressionDatasetGenerator>(nullptr, net, &dataset, nullptr, 10, 1e-4, REG_LOSS_MSE, &out);

	return out;
}

void ann::regression::parameterSearch()
{
	constexpr int minDownsample = 3;
	constexpr int maxDownsample = 10;
	constexpr int minExtra = 0;
	constexpr int maxExtra = 12;

	std::vector<std::pair<double, Configuration>> losses;
	for(int downsample = minDownsample; downsample < maxDownsample; ++downsample)
	{
		for(int extra = minExtra; extra < maxExtra; ++extra)
		{
			Configuration config = {extra, downsample, false};
			Log(Log::INFO, false)<<"Trying ";
			config.print(Log::INFO);
			double loss = tryConfiguration(config);
			losses.push_back({loss, config});

			Log(Log::INFO)<<"Loss for config: "<<loss;
		}
	}

	std::sort(losses.begin(), losses.end(),
		[](std::pair<double, Configuration> a, std::pair<double, Configuration>b){return a.first < b.first;});

	Log(Log::INFO)<<"Best configuration with loss "<<losses.front().first<<'\n';
	losses.front().second.print(Log::INFO);
	Log(Log::INFO)<<"Worst configuration with loss "<<losses.back().first<<'\n';
	losses.back().second.print(Log::INFO);
}*/

