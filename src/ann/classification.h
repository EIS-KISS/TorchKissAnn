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
#include <limits>
#include <memory>
#include <vector>
#include <torch/optim.h>
#include <string>

#include "net.h"
#include "globals.h"
#include "../data/eisdataset.h"
#include "../data/loaders/eisspectradataset.h"
#include "loss/simplicitytolerantloss.h"
#include "log.h"
#include "lossaccumulator.h"
#include "tensoroptions.h"
#include "trainlog.h"
#include "tensoroperators.h"
#include "progressbar.h"
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

inline std::pair<torch::Tensor, torch::Tensor> splitTarget(torch::Tensor targets, bool isMulticlass)
{
	if(isMulticlass || targets.dim() < 2 || targets.size(1) < 2)
		return {targets, torch::Tensor()};
	return {targets.select(1, 0).to(torch::kInt64), targets.narrow(1, 1, targets.size(1)-1)};
}

inline torch::Tensor alignLabelWeights(torch::Tensor weights, int64_t outputSize)
{
	if(weights.size(1) == outputSize)
		return weights;
	if(weights.size(1) > outputSize)
		return weights.narrow(1, 0, outputSize);
	return torch::constant_pad_nd(weights, {0, outputSize - weights.size(1)});
}

/**
 * What NLLLoss divides by with reduction=mean: the sum of the class weights of the targets of this
 * batch, not its number of samples. Weighting a batch by it makes the epoch loss the weighted mean
 * over the whole dataset instead of a mean over batches.
 */
inline double weightedNllBatchWeight(torch::Tensor classWeights, const torch::Tensor& targets)
{
	if(!classWeights.defined() || classWeights.numel() == 0)
		return targets.numel();
	return classWeights.gather(0, targets.reshape({-1}).to(torch::kInt64)).sum().to(torch::kFloat64).template item<double>();
}

template <typename DataLoader>
float trainImpl(std::shared_ptr<Net> network, DataLoader& loader, torch::optim::Optimizer& optimizer,
		   size_t epoch, size_t data_size, torch::Tensor classWeights, bool isMulticlass = false, TrainLog* log = nullptr,
		   const SimplicityTolerantLoss* simplicityLoss = nullptr)
{
	size_t index = 0;
	network->train();
	float accF = 0;
	LossAccumulator lossAcc;

	torch::nn::BCEWithLogitsLoss lossBCE(torch::nn::BCEWithLogitsLossOptions().reduction(torch::kMean).pos_weight(classWeights));
	torch::nn::NLLLoss lossNll(torch::nn::NLLLossOptions().reduction(torch::kMean).weight(classWeights));

	size_t batchesPerPrint = (data_size/batch_size)/1000;
	if(batchesPerPrint == 0)
		batchesPerPrint = 1;

	for(auto& batch : loader)
	{
		torch::Tensor data = network->normalizeInput(batch.data.to(*offload_device));

		std::pair<torch::Tensor, torch::Tensor> label = splitTarget(batch.target.to(*offload_device), isMulticlass);
		torch::Tensor targets = label.first;
		torch::Tensor labelWeights = label.second;

		torch::Tensor prediction = network->forward(data);
		torch::Tensor loss;

		torch::Tensor acc;
		// what this batch counts for in the epoch mean: its samples, or, for a loss that divides by
		// something else than the sample count, exactly what that loss divided by
		double batchWeight = prediction.size(0);

		if(simplicityLoss && labelWeights.defined())
		{
			torch::Tensor aligned = alignLabelWeights(labelWeights, prediction.size(1));
			loss = (*simplicityLoss)(prediction, aligned);
			acc = prediction.argmax(1).eq(targets.reshape({-1})).sum().to(torch::kCPU);
			batchWeight = aligned.sum(1).gt(0).sum().to(torch::kFloat64).template item<double>();
		}
		else if(!isMulticlass)
		{
			targets = targets.reshape({-1});
			loss = lossNll->forward(prediction, targets.to(torch::kInt64));
			acc = prediction.argmax(1).eq(targets).sum().to(torch::kCPU);
			batchWeight = weightedNllBatchWeight(classWeights, targets);
		}
		else
		{
			targets = targets.reshape({prediction.size(0), prediction.size(1)});

			loss = lossBCE->forward(prediction, targets.to(torch::kFloat32)).to(torch::kCPU);
			acc = multiClassHits(torch::sigmoid(prediction), targets, 0.25).sum().to(torch::kCPU);
		}
		assert(!std::isnan(loss.template item<float>()));
		lossAcc.add(loss.template item<float>(), batchWeight);

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
	return static_cast<float>(lossAcc.value());
}

template <typename DataLoader>
torch::Tensor confusion(std::shared_ptr<Net> network, DataLoader& loader)
{
	torch::Tensor confusionMatrix = torch::zeros({network->getOutputSize(), network->getOutputSize()}, tensorOptCpu<long>(false));
	for(const auto& batch : loader)
	{
		torch::Tensor data = network->normalizeInput(batch.data.to(*offload_device));
		torch::Tensor targets = splitTarget(batch.target.to(*offload_device), false).first.reshape({-1});

		torch::Tensor output = network->forward(data);

		targets = unargmax(targets.to(torch::kInt64), network->getOutputSize()).to(torch::kCPU);
		output = unargmax(output.argmax(1), network->getOutputSize()).to(torch::kCPU);

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
	double accOverall;
	double simplicityAccOverall = 0;
};

struct EpochStats
{
	size_t epoch;
	double trainLoss;
	double testLoss;
	double testAccuracy;
	torch::Tensor classAccuracy;
};

template <typename DataLoader>
TestReturn test(std::shared_ptr<Net> network, DataLoader& loader, size_t data_size,
				   int64_t outputSize, torch::Tensor classWeights, bool isMulticlass, size_t epoch = 0,
				   TrainLog* log = nullptr, const SimplicityTolerantLoss* simplicityLoss = nullptr)
{
	network->eval();
	torch::NoGradGuard no_grad;
	LossAccumulator lossAcc;
	float Acc = 0;
	double Acceptable = 0;

	torch::Tensor classAcc = torch::zeros({outputSize}, torch::TensorOptions().dtype(torch::kInt64).device(*offload_device));
	torch::Tensor classCount = torch::zeros({outputSize}, torch::TensorOptions().dtype(torch::kInt64).device(*offload_device));
	torch::Tensor histograms = torch::zeros({outputSize, outputSize}, torch::TensorOptions().dtype(torch::kInt64).device(*offload_device));

	torch::Tensor allpredictions = torch::empty({0}, tensorOptDevice<float>(false));
	torch::Tensor alltargets = torch::empty({0}, tensorOptDevice<float>(false));

	torch::nn::BCEWithLogitsLoss lossBCE(torch::nn::BCEWithLogitsLossOptions().reduction(torch::kMean).weight(classWeights));
	torch::nn::NLLLoss lossNll(torch::nn::NLLLossOptions().reduction(torch::kMean).weight(classWeights));

	std::cout<<"Starting test cycle\n";

	Progress bar("Runing Eval: ", data_size/(batch_size));
	for(const auto& batch : loader)
	{
		bar.tick();
		torch::Tensor data = network->normalizeInput(batch.data.to(*offload_device));

		std::pair<torch::Tensor, torch::Tensor> label = splitTarget(batch.target.to(*offload_device), isMulticlass);
		torch::Tensor targets = label.first;
		torch::Tensor labelWeights = label.second;

		torch::Tensor output = network->forward(data);
		torch::Tensor loss;

		allpredictions = safeConcat({allpredictions, output}, 0);
		alltargets = safeConcat({alltargets, targets.reshape({-1})}, 0);

		// the winner has to be read off before the branches below turn output into something else
		torch::Tensor predicted = output.argmax(1);

		torch::Tensor aligned;
		if(labelWeights.defined())
			aligned = alignLabelWeights(labelWeights, outputSize);

		// what this batch counts for in the mean: its samples, or, for a loss that divides by
		// something else than the sample count, exactly what that loss divided by
		double batchWeight = output.size(0);

		if(simplicityLoss && aligned.defined())
		{
			loss = (*simplicityLoss)(output, aligned);
			batchWeight = aligned.sum(1).gt(0).sum().to(torch::kFloat64).template item<double>();
		}
		else if(!isMulticlass)
		{
			loss = lossNll->forward(output, targets.reshape({-1}).to(torch::kInt64));
			batchWeight = weightedNllBatchWeight(classWeights, targets);
		}
		else
		{
			targets = targets.reshape({output.size(0), output.size(1)});
			loss = lossBCE->forward(output, targets.to(torch::kFloat32));
		}
		assert(!std::isnan(loss.template item<float>()));
		lossAcc.add(loss.template item<float>(), batchWeight);

		if(aligned.defined())
			Acceptable += (aligned.gather(1, predicted.unsqueeze(1)) > 0).sum().to(torch::kFloat64).template item<double>();

		if(!isMulticlass)
		{
			targets = unargmax(targets.reshape({-1}).to(torch::kInt64), outputSize);
			output = unargmax(predicted, outputSize);
		}
		else
		{
			targets = targets.reshape({output.size(0), output.size(1)});
			output = torch::sigmoid(output);
		}

		torch::Tensor hits = multiClassHits(output, targets, 0.25);
		torch::Tensor acc = hits.sum();

		for(int i = 0; i < hits.size(0); ++i)
		{
			if(hits[i].template item<int>() > 0)
				classAcc += targets[i];
			classCount += targets[i];

			histograms[targets.select(0, i).argmax()] = histograms[targets.select(0, i).argmax()] + output.select(0, i);
		}

		Acc += acc.template item<float>();
	}

	bar.markAsCompleted();

	classAcc = classAcc/classCount;
	// the same percent style the per batch train log uses, on the mean over all samples
	double lossMean = lossAcc.value()*100;
	if(log)
	{
		log->logTestLoss(epoch, data_size, lossMean, Acc/data_size, data_size);
		for(int i = 0; i < histograms.size(0); ++i)
			log->logTensor("class_predictions", histograms.select(0, i));
	}

	TestReturn ret;
	ret.targets = alltargets.cpu();
	ret.predictions = allpredictions.cpu();
	ret.histograms = histograms.cpu();
	ret.acc = classAcc.cpu();
	ret.loss = lossMean;
	ret.accOverall = Acc/data_size;
	ret.simplicityAccOverall = data_size > 0 ? Acceptable/static_cast<double>(data_size) : 0;

	return ret;
}

template <typename DatasetType, typename TestDatasetType = DatasetType>
void train(TrainLog* trainLog, std::shared_ptr<Net> net, EisDataset<DatasetType>* trainDataset, EisDataset<TestDatasetType>* testDataset,
		   size_t epochs, double learingRate, bool noWeights, double weightDecay = 0.01, double* finalTestLoss = nullptr,
		   std::vector<EpochStats>* history = nullptr)
{
	assert(net->getOutputSize() == static_cast<int64_t>(trainDataset->outputSize()));
	if(testDataset)
	{
		// The test dataset may cover fewer classes than the network knows; train.cpp remaps its
		// targets onto the training label space by name, so only a superset is impossible here.
		assert(trainDataset->outputSize() >= testDataset->outputSize());
		assert(trainDataset->isMulticlass() == testDataset->isMulticlass());
	}

	net->setPurpose("Classifier");
	net->setExtraInputs(trainDataset->extraInputs());

	if(auto* es = dynamic_cast<EisSpectraDataset*>(trainDataset))
	{
		CanonicalGrid cg;
		if(es->canonicalGridInfo(cg))
			net->setCanonicalGrid(cg.fMin, cg.fMax, (int64_t)cg.points, cg.log);
		SpectraNorm mode = es->spectraNormMode();
		net->setSpectraNorm(mode == SpectraNorm::None ? std::string() : spectraNormToString(mode));
	}

	net->setInputLogCompression(trainDataset->inputSize());
	Log(Log::INFO)<<"Compressing the leading "<<trainDataset->inputSize()<<" input features with a signed log around "<<Net::DEFAULT_INPUT_LOG_FLOOR;

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

	bool simplicityTolerant = trainDataset->hasSimplicityLabels();
	if(simplicityTolerant)
		Log(Log::INFO)<<"Scoring with the simplicity tolerant loss: a spectra is explained by any circuit its label accepts, not only by the one that generated it";

	std::unique_ptr<SimplicityTolerantLoss> simplicityLoss;
	if(simplicityTolerant)
		simplicityLoss.reset(new SimplicityTolerantLoss(classWeights));

	torch::data::DataLoaderOptions options;
	options = options.batch_size(batch_size).workers(JOBS);
	auto trainDataLoader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(trainDataset->map(torch::data::transforms::Stack<>()), options);
	options = options.batch_size(batch_size).workers(JOBS);
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
	torch::optim::AdamW optimizer(net->parameters(), torch::optim::AdamWOptions(learingRate).weight_decay(weightDecay));

	if(trainDataset->isMulticlass())
		Log(Log::DEBUG)<<"Using BCELoss";
	else
		Log(Log::DEBUG)<<"Using NLLLoss";

	double bestTestLoss = std::numeric_limits<double>::infinity();
	for (size_t i = 0; i < epochs; ++i)
	{
		float trainLoss = trainImpl(net, *trainDataLoader, optimizer, i, trainDataset->size().value(),
			classWeights, trainDataset->isMulticlass(), trainLog, simplicityLoss.get());
		bool newBest = false;
		if(testDataset)
		{
			TestReturn ret = test(net, *testDataLoader, testDataset->size().value(), trainDataset->outputSize(),
				classWeights, trainDataset->isMulticlass(), i, trainLog, simplicityLoss.get());
			if(finalTestLoss)
				*finalTestLoss = ret.loss;
			if(history)
				history->push_back(EpochStats{i, static_cast<double>(trainLoss), ret.loss,
					ret.acc.sum().item<double>()/static_cast<double>(ret.acc.numel()), ret.acc});
			if(ret.loss < bestTestLoss)
			{
				bestTestLoss = ret.loss;
				newBest = true;
			}
		}
		else if(history)
			history->push_back(EpochStats{i, static_cast<double>(trainLoss), std::numeric_limits<double>::quiet_NaN(),
				std::numeric_limits<double>::quiet_NaN(), torch::Tensor()});

		if(trainLog)
		{
			std::filesystem::path checkpoint = trainLog->saveNetwork(net);
			if(newBest)
				trainLog->linkBestNetwork(checkpoint);
		}
		Log(Log::INFO)<<"Epoch "<<i<<'/'<<epochs;
	}

	sum = 0;
	for(torch::Tensor& param : net->parameters())
		sum += torch::sum(param).item<double>();
	Log(Log::INFO)<<"End sum "<<sum;

	if(trainLog)
		trainLog->saveNetwork(net, true);
}

}

}
