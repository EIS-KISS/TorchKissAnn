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
#include <ATen/ops/zeros.h>
#include <c10/core/DeviceType.h>
#include <cstddef>
#include <cstdint>
#include <torch/optim.h>
#include <string>

#include "net.h"
#include "globals.h"
#include "../data/eisdataset.h"
#include "log.h"
#include "tensoroptions.h"
#include "trainlog.h"
#include "tensoroperators.h"
#include "indicators.hpp"
#include <ATen/autocast_mode.h>

namespace ann
{

namespace classification
{

torch::Tensor use(torch::Tensor input, std::shared_ptr<Net> net);

torch::Tensor multiClassHits(const torch::Tensor& prediction, torch::Tensor targets, double thresh = 0.5);

void exportRandomSamples(const torch::Tensor& data, const torch::Tensor& targets,
						 const torch::Tensor& hits, const std::filesystem::path& outDir,
						 bool outputProb);

template <typename DataLoader>
void trainImpl(std::shared_ptr<Net> network, DataLoader& loader, torch::optim::Optimizer& optimizer,
		   size_t epoch, size_t data_size, torch::Tensor classWeights, bool isMulticlass = false, TrainLog* log = nullptr)
{
	size_t index = 0;
	network->train();
	float accF = 0;

	torch::nn::BCEWithLogitsLoss lossBCE(torch::nn::BCEWithLogitsLossOptions().reduction(torch::kMean).pos_weight(classWeights));
	torch::nn::NLLLoss lossNll(torch::nn::NLLLossOptions().reduction(torch::kMean).weight(classWeights));

	size_t batchesPerPrint = (data_size/batch_size)/1000;
	if(batchesPerPrint == 0)
		batchesPerPrint = 1;

	for(auto& batch : loader)
	{
		torch::Tensor data = batch.data.to(*offload_device);
		torch::Tensor targets = batch.target.to(*offload_device);

		torch::Tensor prediction = network->forward(data);
		torch::Tensor loss;

		torch::Tensor acc;

		if(!isMulticlass)
		{
			targets = targets.view({-1});
			loss = lossNll->forward(prediction, targets.to(torch::kInt64));
			acc = prediction.argmax(1).eq(targets).sum().to(torch::kCPU);
		}
		else
		{
			targets = targets.reshape({prediction.size(0), prediction.size(1)});

			loss = lossBCE->forward(prediction, targets.to(torch::kFloat32)).to(torch::kCPU);
			acc = multiClassHits(torch::sigmoid(prediction), targets, 0.25).sum().to(torch::kCPU);
		}
		assert(!std::isnan(loss.template item<float>()));

		optimizer.zero_grad();
		loss.backward();
		optimizer.step();

		accF += acc.template item<float>();

		if (index++ % batchesPerPrint == 0 )
		{
			auto end = std::min(data_size, (index + 1) * batch_size);
			if(log)
				log->logTrainLoss(epoch, end, loss.template item<float>()*100, accF/end, data_size);
		}
	}
}

template <typename DataLoader>
torch::Tensor confusion(std::shared_ptr<Net> network, DataLoader& loader)
{
	torch::Tensor confusionMatrix = torch::zeros({network->getOutputSize(), network->getOutputSize()}, tensorOptCpu<long>(false));
	for(const auto& batch : loader)
	{
		torch::Tensor data = batch.data.to(*offload_device);
		torch::Tensor targets = batch.target.to(*offload_device).view({-1});

		torch::Tensor output = network->forward(data);

		targets = unargmax(targets.to(torch::kInt64), network->getOutputSize());
		output = unargmax(output.argmax(1), network->getOutputSize());

		for(long i = 0; i < output.size(0); ++i)
		{
			confusionMatrix[targets.select(0, i).argmax()] = confusionMatrix[targets.select(0, i).argmax()] + output.select(0, i);
		}
	}
	return confusionMatrix;
}

struct TestReturn
{
	torch::Tensor predictions;
	torch::Tensor targets;
	torch::Tensor histograms;
	torch::Tensor acc;
	double loss;
};

template <typename DataLoader>
TestReturn test(std::shared_ptr<Net> network, DataLoader& loader, size_t data_size,
				   int64_t outputSize, torch::Tensor classWeights, bool isMulticlass, size_t epoch = 0,
				   TrainLog* log = nullptr)
{
	network->eval();
	torch::NoGradGuard no_grad;
	float Loss = 0, Acc = 0;

	torch::Tensor classAcc = torch::zeros({outputSize}, torch::TensorOptions().dtype(torch::kInt64).device(*offload_device));
	torch::Tensor classCount = torch::zeros({outputSize}, torch::TensorOptions().dtype(torch::kInt64).device(*offload_device));
	torch::Tensor histograms = torch::zeros({outputSize, outputSize}, torch::TensorOptions().dtype(torch::kInt64).device(*offload_device));

	torch::Tensor allpredictions = torch::empty({0}, tensorOptDevice<float>(false));
	torch::Tensor alltargets = torch::empty({0}, tensorOptDevice<float>(false));

	torch::nn::BCEWithLogitsLoss lossBCE(torch::nn::BCEWithLogitsLossOptions().reduction(torch::kMean).weight(classWeights));
	torch::nn::NLLLoss lossNll(torch::nn::NLLLossOptions().reduction(torch::kMean).weight(classWeights));

	std::cout<<"Starting test cycle\n";

	indicators::BlockProgressBar bar(
		indicators::option::BarWidth(50),
		indicators::option::PrefixText("Runing Eval: "),
		indicators::option::ShowElapsedTime(true),
		indicators::option::ShowRemainingTime(true),
		indicators::option::MaxProgress(data_size/(batch_size*16))
	);
	for(const auto& batch : loader)
	{
		bar.tick();
		torch::Tensor data = batch.data.to(*offload_device);
		torch::Tensor targets = batch.target.to(*offload_device).view({-1});

		at::autocast::set_autocast_enabled(at::kCUDA, true);
		torch::Tensor output = network->forward(data);
		torch::Tensor loss;

		allpredictions = safeConcat({allpredictions, output}, 0);
		alltargets = safeConcat({alltargets, targets}, 0);

		if(!isMulticlass)
		{
			loss = lossNll->forward(output, targets.to(torch::kInt64));
			targets = unargmax(targets.to(torch::kInt64), outputSize);
			output = unargmax(output.argmax(1), outputSize);
		}
		else
		{
			targets = targets.reshape({output.size(0), output.size(1)});
			loss = lossBCE->forward(output, targets.to(torch::kFloat32));
			output = torch::sigmoid(output);
		}
		assert(!std::isnan(loss.template item<float>()));

		torch::Tensor hits = multiClassHits(output, targets, 0.25);
		torch::Tensor acc = hits.sum();

		for(int i = 0; i < hits.size(0); ++i)
		{
			if(hits[i].template item<int>() > 0)
				classAcc += targets[i];
			classCount += targets[i];

			histograms[targets.select(0, i).argmax()] = histograms[targets.select(0, i).argmax()] + output.select(0, i);
		}

		Loss += loss.template item<float>();
		Acc += acc.template item<float>();

		at::autocast::clear_cache();
		at::autocast::set_autocast_enabled(at::kCUDA, false);
	}

	bar.mark_as_completed();

	classAcc = classAcc/classCount;
	if(log)
	{
		log->logTestLoss(epoch, data_size, (Loss / data_size)*100, Acc/data_size, data_size);
		for(int i = 0; i < histograms.size(0); ++i)
			log->logTensor("class_predictions", histograms.select(0, i));
	}

	TestReturn ret;
	ret.targets = alltargets.cpu();
	ret.predictions = allpredictions.cpu();
	ret.histograms = histograms.cpu();
	ret.acc = classAcc.cpu();
	ret.loss = (Loss / data_size)*100;

	return ret;
}

template <typename DatasetType, typename TestDatasetType = DatasetType>
void train(TrainLog* trainLog, std::shared_ptr<Net> net, EisDataset<DatasetType>* trainDataset, EisDataset<TestDatasetType>* testDataset,
		   size_t epochs, double learingRate, bool noWeights)
{
	assert(net->getOutputSize() == static_cast<int64_t>(trainDataset->outputSize()));
	if(testDataset)
	{
		assert(trainDataset->outputSize() == testDataset->outputSize());
		assert(trainDataset->isMulticlass() == testDataset->isMulticlass());
	}

	net->setPurpose("Classifier");
	net->setExtraInputs(trainDataset->extraInputs());
	net->to(*offload_device);

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

	torch::Tensor classWeights;
	if(noWeights)
		classWeights = torch::ones({static_cast<int64_t>(trainDataset->outputSize())}).to(*offload_device);
	else
		classWeights = trainDataset->classWeights().to(*offload_device);

	torch::data::DataLoaderOptions options;
	options = options.batch_size(batch_size).workers(JOBS);
	auto trainDataLoader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(trainDataset->map(torch::data::transforms::Stack<>()), options);
	options = options.batch_size(batch_size*16).workers(JOBS);
	auto testDataLoader = testDataset ? torch::data::make_data_loader(testDataset->map(torch::data::transforms::Stack<>()), options) : nullptr;

	size_t active_parameters = 0;
	size_t inactive_parameters = 0;
	double sum = 0;
	for(torch::Tensor& param : net->parameters())
	{
		if(param.requires_grad())
			active_parameters += param.numel();
		else
			inactive_parameters += param.numel();
		sum += torch::sum(param).item<double>();
	}
	Log(Log::INFO)<<"Training "<<active_parameters<<" active parameters and "<<inactive_parameters<<" inactive parameters sum "<<sum;
	torch::optim::AdamW optimizer(net->parameters(), torch::optim::AdamWOptions(learingRate));

	if(trainDataset->isMulticlass())
		Log(Log::DEBUG)<<"Using BCELoss";
	else
		Log(Log::DEBUG)<<"Using NLLLoss";

	for (size_t i = 0; i < epochs; ++i)
	{
		trainImpl(net, *trainDataLoader, optimizer, i, trainDataset->size().value(),
			classWeights, trainDataset->isMulticlass(), trainLog);
		if(testDataset)
		{
			test(net, *testDataLoader, testDataset->size().value(), trainDataset->outputSize(),
				classWeights, trainDataset->isMulticlass(), i, trainLog);
		}

		trainLog->saveNetwork(net);
		Log(Log::INFO)<<"Epoch "<<i<<'/'<<epochs;
	}

	sum = 0;
	for(torch::Tensor& param : net->parameters())
		sum += torch::sum(param).item<double>();
	Log(Log::INFO)<<"End sum "<<sum;

	trainLog->saveNetwork(net, true);
}

}

}
