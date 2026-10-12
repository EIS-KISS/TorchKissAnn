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

#include <limits>
#include <vector>

#include <torch/nn/modules/loss.h>
#include <torch/optim/adamw.h>
#include <torch/optim/sgd.h>

#include "tensoroperators.h"
#include "ann/autoencoder.h"
#include "autoencoder.h"
#include "globals.h"
#include "../data/eisdataset.h"
#include "data/loaders/eisspectradataset.h"
#include "log.h"
#include "lossaccumulator.h"
#include "tensoroptions.h"
#include "trainlog.h"
#include "progressbar.h"
#include "r2score.h"

namespace ann
{

namespace autoencode
{

torch::Tensor use(torch::Tensor input, std::shared_ptr<AutoEncoder> net, torch::Tensor* latent = nullptr);

inline torch::Tensor klDivergence(const torch::Tensor& mu, const torch::Tensor& logvar)
{
	return torch::mean(-0.5 * torch::sum(1 + logvar - mu.pow(2) - logvar.exp(), 1));
}

inline constexpr double SPECTRAL_MASK_RATIO = 0.2;

// MAE style loss
inline torch::Tensor spectralPredictionLoss(std::shared_ptr<AutoEncoder> network, const torch::Tensor& data,
	int64_t freqCount, int64_t extraCount)
{
	int64_t batch = data.size(0);
	torch::Tensor draw = torch::rand({batch, freqCount}, data.options());
	torch::Tensor holdoutFreq = draw < SPECTRAL_MASK_RATIO;
	torch::Tensor holdoutSpectral = torch::cat({holdoutFreq, holdoutFreq}, 1);
	torch::Tensor holdout = holdoutSpectral;
	if(extraCount > 0)
		holdout = torch::cat({holdoutSpectral, torch::zeros({batch, extraCount}, holdoutSpectral.options())}, 1);
	torch::Tensor holdoutFloat = holdout.to(data.dtype());

	torch::Tensor masked = data * (1.0 - holdoutFloat);
	torch::Tensor prediction = network->forward(masked);
	torch::Tensor squared = (prediction - data).pow(2) * holdoutFloat;
	return squared.sum() / holdoutFloat.sum().clamp(1.0);
}

template <typename DataLoader, typename LossFn>
int trainEpochVae(std::shared_ptr<AutoEncoder> network, DataLoader& loader, LossFn& lossFn, torch::optim::Optimizer& optimizer, double beta, size_t epoch,
			size_t data_size, double spectralWeight, int64_t freqCount, int64_t extraCount, TrainLog* log = nullptr, double* meanTrainLoss = nullptr)
{
	size_t index = 0;
	LossAccumulator lossAcc;
	network->train();

	int64_t loginterval = data_size/10000 ?: (data_size/batch_size-1)/10 ?: 1;

	Progress bar("Runing Eval: ", data_size/(batch_size));

	for(auto& batch : loader)
	{
		torch::Tensor data = network->normalizeInput(batch.data.to(*offload_device));

		torch::Tensor latent;
		torch::Tensor mu;
		torch::Tensor logvar;
		torch::Tensor prediction = network->forward(data, latent, mu, logvar, true);
		torch::Tensor recon = lossFn(prediction, data);
		torch::Tensor kld = klDivergence(mu, logvar);
		torch::Tensor loss = recon + beta * kld;
		if(spectralWeight > 0)
			loss = loss + spectralWeight * spectralPredictionLoss(network, data, freqCount, extraCount);

		if(torch::isnan(torch::sum(loss)).item().to<bool>())
		{
			Log(Log::ERROR)<<loss<<"\nloss contains NAN!";
			return -1;
		}

		optimizer.zero_grad();
		loss.backward();
		optimizer.step();

		lossAcc.add(loss.template item<float>(), data.size(0));

		if(log && index % loginterval == 0)
		{
			auto end = std::min(data_size, (index + 1) * batch_size);
			log->logTrainLoss(epoch, end, loss.template item<float>(), kld.template item<float>(), data_size);
		}

		index++;
	}

	if(meanTrainLoss)
		*meanTrainLoss = lossAcc.value();

	return 0;
}

template <typename DataLoader, typename LossFn>
int trainEpoch(std::shared_ptr<AutoEncoder> network, DataLoader& loader, LossFn& lossFn, torch::optim::Optimizer& optimizer, size_t epoch,
			size_t data_size, double spectralWeight, int64_t freqCount, int64_t extraCount, TrainLog* log = nullptr, double* meanTrainLoss = nullptr)
{
	size_t index = 0;
	LossAccumulator lossAcc;
	network->train();

	int64_t loginterval = data_size/10000 ?: (data_size/batch_size-1)/10 ?: 1;

	torch::Tensor allpredictions = torch::empty({0}, tensorOptDevice<float>(false));
	torch::Tensor alltargets = torch::empty({0}, tensorOptDevice<float>(false));
	torch::Tensor alltargetmse = torch::zeros({network->getOutputSize()}, tensorOptDevice<float>(false));

	Progress bar("Runing Eval: ", data_size/(batch_size));

	torch::Tensor loss;

	for(auto& batch : loader)
	{
		torch::Tensor data = network->normalizeInput(batch.data.to(*offload_device));

		torch::Tensor prediction = network->forward(data);
		torch::Tensor loss = lossFn(prediction, data);
		if(spectralWeight > 0)
			loss = loss + spectralWeight * spectralPredictionLoss(network, data, freqCount, extraCount);

		if(torch::isnan(torch::sum(loss)).item().to<bool>())
		{
			Log(Log::ERROR)<<loss<<"\nloss contains NAN!";
			return -1;
		}

		optimizer.zero_grad();
		loss.backward();
		optimizer.step();

		lossAcc.add(loss.template item<float>(), data.size(0));

		if(log && index % loginterval == 0)
		{
			auto end = std::min(data_size, (index + 1) * batch_size);
			log->logTrainLoss(epoch, end, loss.template item<float>(), 0, data_size);
		}

		index++;
	}

	if(meanTrainLoss)
		*meanTrainLoss = lossAcc.value();

	return 0;
}

struct TestReturn
{
	torch::Tensor latents;
	torch::Tensor datasetTargets;
	torch::Tensor outputs;
	torch::Tensor inputs;

	torch::Tensor mse;
	double loss;
};

struct EpochStats
{
	size_t epoch;
	double trainLoss;
	double testLoss;
	torch::Tensor mse;
};

template <typename DataLoader, typename LossFn>
TestReturn test(std::shared_ptr<AutoEncoder> network, DataLoader& loader, LossFn& lossFn, size_t data_size, size_t epoch = 0, TrainLog* log = nullptr)
{
	network->eval();
	torch::NoGradGuard noGrad;

	LossAccumulator lossAcc;

	torch::Tensor alllatets = torch::empty({0}, tensorOptDevice<float>(false));
	torch::Tensor alldatasettargets = torch::empty({0}, tensorOptDevice<float>(false));
	torch::Tensor alloutputs = torch::empty({0}, tensorOptDevice<float>(false));
	torch::Tensor allinputs = torch::empty({0}, tensorOptDevice<float>(false));
	torch::Tensor alltargetmse = torch::zeros({network->getOutputSize()}, tensorOptDevice<float>(false));

	Progress bar("Runing Eval: ", data_size/(batch_size));

	for(const auto& batch : loader)
	{
		bar.tick();
		torch::Tensor data = network->normalizeInput(batch.data.to(*offload_device));

		torch::Tensor latent;
		torch::Tensor output = network->forward(data, latent);

		alllatets = safeConcat({alllatets, latent}, 0);
		alldatasettargets = safeConcat({alldatasettargets, batch.target}, 0);
		alloutputs = safeConcat({alloutputs, output}, 0);
		allinputs = safeConcat({allinputs, data}, 0);

		torch::Tensor loss = lossFn(output, data);
		torch::Tensor mse = torch::nn::functional::mse_loss(output, data, torch::nn::functional::MSELossFuncOptions(torch::kNone));
		mse = torch::mean(mse, 0);
		alltargetmse = alltargetmse + mse*data.size(0);

		lossAcc.add(loss.template item<float>(), data.size(0));
	}

	bar.markAsCompleted();

	alltargetmse = alltargetmse / lossAcc.count();

	TestReturn ret;
	ret.loss = lossAcc.value();
	ret.latents = alllatets.cpu();
	ret.mse = alltargetmse.cpu();
	ret.datasetTargets = alldatasettargets.cpu();
	ret.outputs = alloutputs.cpu();
	ret.inputs = allinputs.cpu();

	if(log)
		log->logTestLoss(epoch, 0, ret.loss, ret.mse.mean().item().toDouble());

	return ret;
}

template <typename DatasetType, typename TestDatasetType = DatasetType>
void train(TrainLog* trainLog, std::shared_ptr<AutoEncoder> net, EisDataset<DatasetType>* dataset, EisDataset<TestDatasetType>* testDataset, size_t epochs, double learingRate, double beta = 1.0,
	double weightDecay = 0, double* finalTestLoss = nullptr, std::vector<EpochStats>* history = nullptr, double spectralWeight = 0, SpectraNorm spectraNorm = SpectraNorm::None)
{
	assert(net->getInputSize() == dataset->get(0).data.size(0));
	if(testDataset)
		assert(net->getInputSize() == testDataset->get(0).data.size(0));
	else
		Log(Log::WARN)<<"No test dataset provided!";

	if(net->isVariational())
		net->setPurpose("Autoencoder,VAE");
	else
		net->setPurpose("Autoencoder");
	net->setExtraInputs(dataset->extraInputs());

	if(auto* es = dynamic_cast<EisSpectraDataset*>(dataset))
	{
		CanonicalGrid cg;
		if(es->canonicalGridInfo(cg))
			net->setCanonicalGrid(cg.fMin, cg.fMax, (int64_t)cg.points, cg.log);
	}

	net->setSpectraNorm(spectraNorm == SpectraNorm::None ? std::string() : spectraNormToString(spectraNorm));

	net->setInputLogCompression(dataset->inputSize());
	Log(Log::INFO)<<"Compressing the leading "<<dataset->inputSize()<<" input features with a signed log around "<<Net::DEFAULT_INPUT_LOG_FLOOR;

	net->to(*offload_device);

	torch::data::DataLoaderOptions options;
	options = options.batch_size(batch_size).workers(JOBS);
	auto trainDataLoader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(dataset->map(torch::data::transforms::Stack<>()), options);
	auto testDataLoader = testDataset ? torch::data::make_data_loader<torch::data::samplers::RandomSampler>(testDataset->map(torch::data::transforms::Stack<>()), options) : nullptr;

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

	torch::nn::MSELoss mseloss(torch::nn::MSELossOptions().reduction(torch::kMean));

	// ramp the KL weight in linearly over the first epochs so the divergence term can not dominate the reconstruction gradient while encoder is still random
	size_t klWarmup = std::min<size_t>(epochs, 10);

	int64_t extraCount = 0;
	for(const auto& extra : dataset->extraInputs())
		extraCount += extra.second;
	int64_t spectralFeatures = net->getInputSize() - extraCount;
	int64_t freqCount = spectralFeatures / 2;
	if(spectralWeight > 0 && (spectralFeatures < 4 || spectralFeatures % 2 != 0))
	{
		Log(Log::WARN)<<"spectral prediction loss disabled: the "<<net->getInputSize()<<" wide input is not a real/imaginary spectra";
		spectralWeight = 0;
	}
	if(spectralWeight > 0)
		Log(Log::INFO)<<"Spectral prediction auxiliary loss active: weight "<<spectralWeight<<", holding out "<<(SPECTRAL_MASK_RATIO*100)<<"% of "<<freqCount<<" frequency points per sample";

	for (size_t i = 0; i < epochs; ++i)
	{
		int epochRet;
		double meanTrainLoss = std::numeric_limits<double>::quiet_NaN();
		if(net->isVariational())
		{
			double epochBeta = beta;
			if(klWarmup > 0)
				epochBeta = beta * std::min(1.0, double(i + 1) / double(klWarmup));
			epochRet = trainEpochVae(net, *trainDataLoader, mseloss, optimizer, epochBeta, i, dataset->size().value(), spectralWeight, freqCount, extraCount, trainLog, &meanTrainLoss);
		}
		else
			epochRet = trainEpoch(net, *trainDataLoader, mseloss, optimizer, i, dataset->size().value(), spectralWeight, freqCount, extraCount, trainLog, &meanTrainLoss);

		if(epochRet != 0)
			break;

		if(testDataset)
		{
			TestReturn testret = test(net, *testDataLoader, mseloss, dataset->size().value(), i, trainLog);
			Log(Log::INFO)<<"Mse: "<<testret.mse;
			if(finalTestLoss)
				*finalTestLoss = testret.loss;
			if(history)
				history->push_back(EpochStats{i, meanTrainLoss, testret.loss, testret.mse});
		}
		else if(history)
			history->push_back(EpochStats{i, meanTrainLoss, std::numeric_limits<double>::quiet_NaN(), torch::Tensor()});

		if(trainLog)
			trainLog->saveNetwork(net);
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
