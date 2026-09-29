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
#include <format>
#include <limits>
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
#include "loss/permutationmsseloss.h"
#include "loss/eiscompositeloss.h"
#include "data/regressiondataset.h"
#include "data/loaders/eisspectradataset.h"
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
	REG_LOSS_MSE,
	REG_LOSS_PERMUTATION,
	REG_LOSS_COMPOSITE
} regression_loss_type_t;

torch::Tensor use(torch::Tensor input, std::shared_ptr<Net> net);

template <typename DatasetType>
void train(std::string fileName, EisDataset<DatasetType>* trainDataset, bool convNet, size_t epochs = 30);

std::pair<torch::Tensor, torch::Tensor> getTargetScalesAndBiases(const std::string& modelString);

template <typename DataLoader, typename LossFn>
int trainImpl(std::shared_ptr<Net> network, DataLoader& loader,
			LossFn& lossFn, torch::optim::Optimizer& optimizer, size_t epoch,
			size_t data_size, int64_t outputSize, TrainLog* log = nullptr, double* finalLoss = nullptr,
			float* meanTrainLoss = nullptr)
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
		torch::Tensor data = network->normalizeInput(batch.data.to(*offload_device));
		torch::Tensor targets = network->normalizeTarget(batch.target.to(*offload_device));

		torch::Tensor prediction = network->forward(data);

		parameters.push_back(network->named_parameters(true));
		for(torch::OrderedDict<std::string, torch::Tensor>::Item& parameter : parameters.back())
			parameter.value() = parameter.value().clone();
		if(parameters.size() > 3)
			parameters.erase(parameters.begin());

		if(!torch::isfinite(prediction).all().item().to<bool>())
		{
			for(torch::OrderedDict<std::string, torch::Tensor> dict : parameters)
			{
				Log(Log::ERROR)<<"STEP";
				for(torch::OrderedDict<std::string, torch::Tensor>::Item& parameter : dict)
				{
					Log(Log::ERROR)<<parameter.key()<<'\n'<<parameter.value();
				}
			}
			Log(Log::ERROR)<<prediction<<"\nPrediction contains NaN or Inf!\nData:\n"<<data;
			return -1;
		}

		torch::Tensor predictionOrig = prediction.clone();
		torch::Tensor base = torch::ones(prediction.sizes(), tensorOptCpu<float>().device(*offload_device))*10;
		if(!torch::isfinite(prediction).all().item().to<bool>())
		{
			Log(Log::ERROR)<<prediction<<"\nPrediction after scaleing contains NaN or Inf!\nData:\n"<<data;
			return -1;
		}
		loss = lossFn(prediction, targets);
		/*torch::Tensor negatives = ((torch::sign(prediction)-1)*-1)/2;
		loss = loss + (negatives.sum()/negatives.numel());*/

		if(!torch::isfinite(loss).all().item().to<bool>())
		{
			Log(Log::ERROR)<<loss<<"\nloss contains NaN or Inf!";
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

	if(meanTrainLoss)
		*meanTrainLoss = index ? (lossAccumulator/index) : std::numeric_limits<float>::quiet_NaN();

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

struct EpochStats
{
	size_t epoch;
	double trainLoss;
	double testLoss;
	double r2Mean;
	torch::Tensor r2;
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
		torch::Tensor data = network->normalizeInput(batch.data.to(*offload_device));
		torch::Tensor targets = network->normalizeTarget(batch.target.to(*offload_device));

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
	// the r2 and mse above are computed in the normalized space the network works in, the predictions and targets handed out here are physical values again
	ret.predictions = network->denormalizeOutput(allpredictions).cpu();
	ret.targets = network->denormalizeOutput(alltargets).cpu();
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
		   size_t epochs, double learingRate, regression_loss_type_t loss = REG_LOSS_MSE, double* finalLoss = nullptr,
		   double weightDecay = 0.001, double* finalTestLoss = nullptr, std::vector<EpochStats>* history = nullptr)
{
	static constexpr double LOSS_START_DECADE = -2;
	static constexpr double LOSS_FINISH_DECADE = 5;
	static constexpr int64_t LOSS_POINT_COUNT = 32;

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

	RegressionNormalization norm = trainDataset->getNormalization();
	Log(Log::INFO)<<"Using output scalars:\n"<<norm.targetScalars;
	Log(Log::INFO)<<"Using output biases:\n"<<norm.targetBiases;
	Log(Log::INFO)<<"Learning the outputs flagged here in the log domain:\n"<<norm.targetLogMask;
	Log(Log::INFO)<<"Using input scalars:\n"<<norm.inputScalars;
	Log(Log::INFO)<<"Using input biases:\n"<<norm.inputBiases;
	Log(Log::INFO)<<"Compressing the leading "<<norm.inputLogCount<<" input features with a signed log around "<<std::format("{:g}", norm.inputLogFloor);
	net->setOutputScalars(norm.targetScalars);
	net->setOutputBiases(norm.targetBiases);
	net->setOutputLogMask(norm.targetLogMask);
	net->setInputNormalization(norm.inputScalars, norm.inputBiases, norm.inputLogCount, norm.inputLogFloor);

	net->setPurpose(purposeString(trainDataset));
	net->setExtraInputs(trainDataset->extraInputs());
	net->setInputFrequencies(trainDataset->frequencies().value());

	// Record the canonical frequency grid (if any) so the checkpoint and the ONNX export carry
	// everything libkissinference needs to resample a raw spectra before feeding the fixed graph.
	{
		CanonicalGrid cg;
		if(auto* es = dynamic_cast<EisSpectraDataset*>(trainDataset))
			if(es->canonicalGridInfo(cg))
				net->setCanonicalGrid(cg.fMin, cg.fMax, (int64_t)cg.points, cg.log);
	}

	net->to(*offload_device);

	Log(Log::INFO)<<"Training "<<net->getPurpose();

	std::unique_ptr<torch::nn::MSELoss> lossMse;
	std::unique_ptr<EisDistanceLoss> lossEis;
	std::unique_ptr<PermutationMseLoss> lossPermutation;
	std::unique_ptr<EisCompositeLoss> lossComposite;
	if(loss == REG_LOSS_EIS || loss == REG_LOSS_PERMUTATION || loss == REG_LOSS_COMPOSITE)
	{
		torch::Tensor omega = trainDataset->frequencies().value_or(torch::Tensor());
		if(omega.numel() < 2)
		{
			Log(Log::WARN)<<"Dataset does not provide frequencies, simulating the loss on a fallback grid";
			omega = torch::logspace(LOSS_START_DECADE, LOSS_FINISH_DECADE, LOSS_POINT_COUNT);
		}
		if(loss == REG_LOSS_EIS)
		{
			Log(Log::INFO)<<"Loss will simulate "<<omega.numel()<<" points from "<<omega.min().item<float>()
				<<" to "<<omega.max().item<float>()<<" rad/s and will be based on "<<trainDataset->targetName();
			lossEis.reset(new EisDistanceLoss(trainDataset->targetName(), omega,
						norm.targetScalars, norm.targetBiases, norm.targetLogMask));
		}
		else if(loss == REG_LOSS_PERMUTATION)
		{
			Log(Log::INFO)<<"Loss will score the predictions in the normalized parameter space against the equivalent parameter permutation with the smallest error on "
				<<trainDataset->targetName();
			lossPermutation.reset(new PermutationMseLoss(trainDataset->targetName(), omega,
						norm.targetScalars, norm.targetBiases, norm.targetLogMask));
		}
		else
		{
			Log(Log::INFO)<<"Loss will combine the spectra distance with the equivalent parameter permutation score on "
				<<trainDataset->targetName();
			lossComposite.reset(new EisCompositeLoss(trainDataset->targetName(), omega,
						norm.targetScalars, norm.targetBiases, norm.targetLogMask));
		}
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
	torch::optim::AdamW optimizer(net->parameters(), torch::optim::AdamWOptions(learingRate).weight_decay(weightDecay));

	double bestTestLoss = std::numeric_limits<double>::infinity();
	for (size_t epoch = 0; epoch < epochs; ++epoch)
	{
		int ret;
		float meanTrainLoss = std::numeric_limits<float>::quiet_NaN();
		if(lossEis)
			 ret = trainImpl(net, *trainDataLoader,
				*lossEis, optimizer, epoch, trainDataset->size().value(),
				trainDataset->outputSize(), trainLog, finalLoss, &meanTrainLoss);
		else if(lossPermutation)
			 ret = trainImpl(net, *trainDataLoader,
				*lossPermutation, optimizer, epoch, trainDataset->size().value(),
				trainDataset->outputSize(), trainLog, finalLoss, &meanTrainLoss);
		else if(lossComposite)
			 ret = trainImpl(net, *trainDataLoader,
				*lossComposite, optimizer, epoch, trainDataset->size().value(),
				trainDataset->outputSize(), trainLog, finalLoss, &meanTrainLoss);
		else
			ret = trainImpl(net, *trainDataLoader,
							*lossMse, optimizer, epoch, trainDataset->size().value(),
							trainDataset->outputSize(), trainLog, finalLoss, &meanTrainLoss);
		if(ret != 0)
			return -1;

		bool newBest = false;
		if(testDataset)
		{
			// evaluate with the same loss the network was trained with
			TestReturn testret = lossEis ?
				test(net, *testDataLoader, *lossEis, testDataset->size().value(), epoch, trainLog) :
				lossPermutation ?
				test(net, *testDataLoader, *lossPermutation, testDataset->size().value(), epoch, trainLog) :
				lossComposite ?
				test(net, *testDataLoader, *lossComposite, testDataset->size().value(), epoch, trainLog) :
				test(net, *testDataLoader, *lossMse, testDataset->size().value(), epoch, trainLog);
			Log(Log::INFO)<<"Test r2: "<<testret.r2;
			Log(Log::INFO)<<"Test mse: "<<testret.mse;
			if(finalTestLoss)
				*finalTestLoss = testret.loss;
			if(history)
				history->push_back(EpochStats{epoch, static_cast<double>(meanTrainLoss), testret.loss,
					testret.r2.mean().item<double>(), testret.r2});
			if(testret.loss < bestTestLoss)
			{
				bestTestLoss = testret.loss;
				newBest = true;
			}
		}
		else if(history)
			history->push_back(EpochStats{epoch, static_cast<double>(meanTrainLoss),
				std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(),
				torch::Tensor()});


		if(trainLog)
		{
			std::filesystem::path checkpoint = trainLog->saveNetwork(net);
			if(newBest)
				trainLog->linkBestNetwork(checkpoint);
		}
		Log(Log::INFO)<<"Epoch "<<epoch<<'/'<<epochs;
	}

	if(trainLog)
		trainLog->saveNetwork(net, true);
	return 0;
}


}
}
