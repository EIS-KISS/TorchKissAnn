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
#include <ATen/ops/logspace.h>
#include <cstddef>
#include <cstdint>
#include <torch/csrc/autograd/anomaly_mode.h>
#include <torch/enum.h>
#include <torch/nn/functional/loss.h>
#include <torch/nn/modules/loss.h>
#include <torch/nn/options/loss.h>
#include <torch/optim.h>
#include <string>
#include <eisgenerator/model.h>
#include <torch/optim/adamw.h>
#include <torch/optim/sgd.h>
#include <torch/ordered_dict.h>

#include "net.h"
#include "globals.h"
#include "log.h"
#include "trainlog.h"
#include "loss/eisdistanceloss.h"
#include "data/regressiondataset.h"
#include "tensoroptions.h"
#include "tensoroperators.h"
#include "r2score.h"

namespace ann
{
namespace regression
{

typedef enum
{
	REG_LOSS_EIS,
	REG_LOSS_MSE
} regression_loss_type_t;

torch::Tensor use(torch::Tensor input, std::shared_ptr<Net> net);

template <typename DatasetType>
void train(std::string fileName, EisDataset<DatasetType>* trainDataset, bool convNet, size_t epochs = 30);

std::pair<torch::Tensor, torch::Tensor> getTargetScalesAndBiases(const std::string& modelString);

template <typename DataLoader, typename LossFn>
int trainImpl(std::shared_ptr<Net> network, DataLoader& loader,
			LossFn& lossFn, torch::optim::Optimizer& optimizer, size_t epoch,
			size_t data_size, int64_t outputSize, TrainLog* log = nullptr, double* finalLoss = nullptr)
{
	size_t index = 0;
	network->train();

	float lossAccumulator = 0;
	torch::Tensor distanceAccumulator = torch::zeros({outputSize}, torch::TensorOptions().dtype(torch::kFloat32).device(*offload_device));

	int64_t loginterval = data_size/10000 ?: (data_size/batch_size-1)/10 ?: 1;

	std::vector<torch::OrderedDict<std::string, torch::Tensor>> parameters;

	torch::Tensor prevLoss;
	torch::Tensor loss;

	for(auto& batch : loader)
	{
		torch::Tensor data = batch.data.to(*offload_device);
		torch::Tensor targets = ((batch.target-network->getOutputBiases())/network->getOutputScalars()).to(*offload_device);

		torch::Tensor prediction = network->forward(data);

		parameters.push_back(network->named_parameters(true));
		for(torch::OrderedDict<std::string, torch::Tensor>::Item& parameter : parameters.back())
			parameter.value() = parameter.value().clone();
		if(parameters.size() > 3)
			parameters.erase(parameters.begin());

		if(torch::isnan(torch::sum(prediction)).item().to<bool>())
		{
			for(torch::OrderedDict<std::string, torch::Tensor> dict : parameters)
			{
				Log(Log::ERROR)<<"STEP";
				for(torch::OrderedDict<std::string, torch::Tensor>::Item& parameter : dict)
				{
					Log(Log::ERROR)<<parameter.key()<<'\n'<<parameter.value();
				}
			}
			Log(Log::ERROR)<<prediction<<"\nPrediction contains NAN!\nData:\n"<<data;
			return -1;
		}

		torch::Tensor predictionOrig = prediction.clone();
		torch::Tensor base = torch::ones(prediction.sizes(), tensorOptCpu<float>().device(*offload_device))*10;
		if(torch::isnan(torch::sum(prediction)).item().to<bool>())
		{
			Log(Log::ERROR)<<prediction<<"\nPrediction after scaleing contains NAN!\nData:\n"<<data;
			return -1;
		}
		loss = lossFn(prediction, targets);
		/*torch::Tensor negatives = ((torch::sign(prediction)-1)*-1)/2;
		loss = loss + (negatives.sum()/negatives.numel());*/

		if(torch::isnan(torch::sum(loss)).item().to<bool>())
		{
			Log(Log::ERROR)<<loss<<"\nloss contains NAN!";
			Log(Log::DEBUG)<<"targets:\n"<<targets<<'\n';
			Log(Log::DEBUG)<<"prediction:\n"<<predictionOrig<<'\n';
			Log(Log::DEBUG)<<"prediction clamped:\n"<<predictionOrig.clamp(-12, 7)<<'\n';
			Log(Log::DEBUG)<<"prediction pow:\n"<<prediction<<'\n';
			Log(Log::DEBUG)<<"loss:\n"<<loss<<'\n';
			return -1;
		}

		optimizer.zero_grad();
		loss.backward();
		optimizer.step();

		lossAccumulator += loss.template item<float>();

		if(log && index % loginterval == 0)
		{
			//Log(Log::INFO)<<"last targets:\n"<<targets[0];
			//Log(Log::INFO)<<"last prediction:\n"<<prediction[0];
			auto end = std::min(data_size, (index + 1) * batch_size);
			log->logTrainLoss(epoch, end, loss.template item<float>(), 0, data_size);
		}

		prevLoss = loss;

		index++;
	}

	if(log)
		log->logTrainLoss(epoch, data_size, lossAccumulator/index, 0, data_size);

	if(finalLoss)
		*finalLoss = loss.item().toDouble();

	return 0;
}

struct TestReturn
{
	torch::Tensor predictions;
	torch::Tensor targets;

	torch::Tensor mse;
	torch::Tensor r2;
	double loss;
};

template <typename DataLoader, typename LossFn>
TestReturn test(std::shared_ptr<Net> network, DataLoader& loader, LossFn& lossFn, size_t data_size, size_t epoch = 0, TrainLog* log = nullptr)
{
	network->eval();
	torch::NoGradGuard noGrad;

	float lossAccumulator = 0;

	torch::Tensor allpredictions = torch::empty({0}, tensorOptDevice<float>(false));
	torch::Tensor alltargets = torch::empty({0}, tensorOptDevice<float>(false));
	torch::Tensor alltargetmse = torch::zeros({network->getOutputSize()}, tensorOptDevice<float>(false));

	indicators::BlockProgressBar bar(
		indicators::option::BarWidth(50),
		indicators::option::PrefixText("Runing Eval: "),
		indicators::option::ShowElapsedTime(true),
		indicators::option::ShowRemainingTime(true),
		indicators::option::MaxProgress(data_size/(batch_size))
	);

	size_t index = 0;
	for(const auto& batch : loader)
	{
		bar.tick();
		torch::Tensor data = batch.data.to(*offload_device);
		torch::Tensor targets = ((batch.target-network->getOutputBiases())/network->getOutputScalars()).to(*offload_device);

		torch::Tensor output = network->forward(data);

		allpredictions = safeConcat({allpredictions, output}, 0);
		alltargets = safeConcat({alltargets, targets}, 0);

		torch::Tensor loss = lossFn(output, targets);
		torch::Tensor mse = torch::nn::functional::mse_loss(output, targets, torch::nn::functional::MSELossFuncOptions(torch::kNone));
		mse = torch::mean(mse, 0);
		alltargetmse = alltargetmse + mse;

		lossAccumulator += loss.template item<float>();
		index++;
	}

	bar.mark_as_completed();

	alltargetmse = alltargetmse / index;

	torch::Tensor r2scores = r2score(allpredictions, alltargets);
	if(log)
		log->logTestLoss(epoch, data_size, (lossAccumulator / index), 0, data_size);

	TestReturn ret;
	ret.loss = (lossAccumulator / index);
	ret.predictions = allpredictions.cpu()*network->getOutputScalars() + network->getOutputBiases();
	ret.targets = alltargets.cpu()*network->getOutputScalars() + network->getOutputBiases();
	ret.mse = alltargetmse.cpu();
	ret.r2 = r2scores.cpu();

	return ret;
}

template <typename DatasetType>
std::string purposeString(RegressionDataset<DatasetType>* dataset)
{
	std::stringstream purpose;
	std::string regressionTarget = dataset->targetName();
	purpose<<"Regression,"<<regressionTarget;
	return purpose.str();
}

template <typename DatasetType, typename TestDatasetType = DatasetType>
int train(TrainLog* trainLog, std::shared_ptr<Net> net, RegressionDataset<DatasetType>* trainDataset, RegressionDataset<TestDatasetType>* testDataset,
		   size_t epochs, double learingRate, regression_loss_type_t loss = REG_LOSS_MSE, double* finalLoss = nullptr)
{
	static constexpr double LOSS_START_DECADE = -2;
	static constexpr double LOSS_FINISH_DECADE = 6;
	static constexpr int64_t LOSS_POINT_COUNT = 3;

	assert(net->getOutputSize() == static_cast<int64_t>(trainDataset->outputSize()));
	assert(net->getInputSize() == trainDataset->get(0).data.size(0));
	assert(trainDataset->isMulticlass());
	assert(!net->hasSoftmaxOutput());
	if(testDataset)
	{
		assert(trainDataset->outputSize() == testDataset->outputSize());
		assert(testDataset->isMulticlass());
	}
	else
	{
		Log(Log::WARN)<<"No test dataset provided!";
	}

	std::pair<torch::Tensor, torch::Tensor> scalarAndBias = trainDataset->getTargetScalesAndBias();
	Log(Log::INFO)<<"Using output scalars:\n"<<scalarAndBias.first;
	Log(Log::INFO)<<"Using output biases:\n"<<scalarAndBias.second;
	net->setOutputScalars(scalarAndBias.first);
	net->setOutputBiases(scalarAndBias.second);

	net->setPurpose(purposeString(trainDataset));
	net->setExtraInputs(trainDataset->extraInputs());
	net->setInputFrequencies(trainDataset->frequencies().value());

	net->to(*offload_device);

	Log(Log::INFO)<<"Training "<<net->getPurpose();

	std::unique_ptr<torch::nn::MSELoss> lossMse;
	std::unique_ptr<EisDistanceLoss> lossEis;
	if(loss == REG_LOSS_EIS)
	{
		Log(Log::INFO)<<"Loss will simulate from 10^"<<LOSS_START_DECADE
			<<" to 10^"<<LOSS_FINISH_DECADE<<" with "<<LOSS_POINT_COUNT<<" steps "
			<<"and will be based on "<<trainDataset->targetName();
		torch::Tensor omega = torch::logspace(LOSS_START_DECADE, LOSS_FINISH_DECADE, LOSS_POINT_COUNT);
		lossEis.reset(new EisDistanceLoss(trainDataset->targetName(), omega));
	}
	else
	{
		Log(Log::INFO)<<"Using mse loss";
		lossMse.reset(new torch::nn::MSELoss(torch::nn::MSELossOptions().reduction(torch::kMean)));
	}

	std::vector<std::string> outputLables(net->getOutputSize());
	bool noLabels = false;
	for(int64_t i = 0; i < net->getOutputSize(); ++i)
	{
		std::string label = trainDataset->outputName(i);
		if(label == "Unkown")
		{
			Log(Log::WARN)<<"This dataset dosent provide output labels";
			noLabels = true;
			break;
		}
		outputLables[i] = label;
	}
	if(!noLabels)
		net->setOutputLabels(outputLables);

	torch::data::DataLoaderOptions options;
	options = options.batch_size(batch_size).workers(1);
	auto trainDataLoader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>
		(trainDataset->map(torch::data::transforms::Stack<>()), options);
	//options.batch_size(5);
	auto testDataLoader =
		testDataset ? torch::data::make_data_loader(testDataset->map(torch::data::transforms::Stack<>()), options) : nullptr;
	torch::optim::AdamW optimizer(net->parameters(), torch::optim::AdamWOptions(learingRate).weight_decay(0.001));

	for (size_t epoch = 0; epoch < epochs; ++epoch)
	{
		int ret;
		if(lossEis)
			 ret = trainImpl(net, *trainDataLoader,
				*lossEis, optimizer, epoch, trainDataset->size().value(),
				trainDataset->outputSize(), trainLog, finalLoss);
		else
			ret = trainImpl(net, *trainDataLoader,
							*lossMse, optimizer, epoch, trainDataset->size().value(),
							trainDataset->outputSize(), trainLog, finalLoss);
		if(ret != 0)
			return -1;

		if(testDataset)
		{
			TestReturn testret = test(net, *testDataLoader, *lossMse, testDataset->size().value(), epoch, trainLog);
			Log(Log::INFO)<<"Test r2: "<<testret.r2;
			Log(Log::INFO)<<"Test mse: "<<testret.mse;
		}


		if(trainLog)
			trainLog->saveNetwork(net);
		Log(Log::INFO)<<"Epoch "<<epoch<<'/'<<epochs;
	}

	if(trainLog)
		trainLog->saveNetwork(net, true);
	return 0;
}


}
}
