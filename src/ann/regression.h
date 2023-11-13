#pragma once
#include <ATen/ops/logspace.h>
#include <cstddef>
#include <cstdint>
#include <torch/csrc/autograd/anomaly_mode.h>
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
#include "data/eisdataset.h"
#include "tensoroptions.h"
#include "tensoroperators.h"

namespace ann
{
namespace regression
{

torch::Tensor use(torch::Tensor input, std::shared_ptr<Net> net);

template <typename DatasetType>
void train(std::string fileName, EisDataset<DatasetType>* trainDataset, bool convNet, size_t epochs = 30);

std::pair<torch::Tensor, torch::Tensor> getTargetScalesAndBiases(const std::string& modelString);

template <typename DataLoader, typename LossFn>
int trainImpl(std::shared_ptr<Net> network, DataLoader& loader,
			LossFn& lossFn, torch::optim::Optimizer& optimizer, size_t epoch,
			size_t data_size, int64_t classesCount, TrainLog* log = nullptr)
{
	size_t index = 0;
	network->train();

	float lossAccumulator = 0;
	torch::Tensor distanceAccumulator = torch::zeros({classesCount}, torch::TensorOptions().dtype(torch::kFloat32).device(*offload_device));

	int64_t loginterval = data_size/10000 ?: data_size/batch_size-1;

	std::vector<torch::OrderedDict<std::string, torch::Tensor>> parameters;

	torch::Tensor prevLoss;

	for(auto& batch : loader)
	{
		torch::Tensor data = batch.data.to(*offload_device);
		torch::Tensor targets = (batch.target.to(*offload_device)-network->getOutputBiases())/network->getOutputScalars();

		torch::Tensor prediction = network->forward(data)+0.5;

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

		torch::Tensor loss = lossFn(prediction, targets);

		if(torch::isnan(torch::sum(loss)).item().to<bool>())
		{
			Log(Log::ERROR)<<loss<<"\nloss contains NAN!";
			Log(Log::DEBUG)<<"targets:\n"<<targets<<'\n';
			Log(Log::DEBUG)<<"prediction:\n"<<predictionOrig<<'\n';
			Log(Log::DEBUG)<<"prediction clamped:\n"<<predictionOrig.clamp(-12, 7)<<'\n';
			Log(Log::DEBUG)<<"prediction pow:\n"<<prediction<<'\n';
			Log(Log::DEBUG)<<"loss:\n"<<loss<<'\n';
			saveTensorForPytorch("pred.zip", predictionOrig);
			return -1;
		}

		optimizer.zero_grad();
		loss.backward();
		optimizer.step();

		lossAccumulator += loss.template item<float>();

		if(log && index % loginterval == 0)
		{
			Log(Log::INFO)<<"last targets:\n"<<targets[0];
			Log(Log::INFO)<<"last prediction:\n"<<prediction[0];
			auto end = std::min(data_size, (index + 1) * batch_size);
			log->logTrainLoss(epoch, end, loss.template item<float>(), 0, data_size);
		}

		prevLoss = loss;

		index++;
	}

	log->logTrainLoss(epoch, data_size, lossAccumulator/index, 0, data_size);

	return 0;
}

template <typename DataLoader, typename LossFn>
torch::Tensor test(std::shared_ptr<Net> network, DataLoader& loader, LossFn& lossFn, size_t data_size,
				   int64_t classesCount, size_t epoch = 0, TrainLog* log = nullptr)
{
	network->eval();
	torch::NoGradGuard noGrad;

	float lossAccumulator = 0;
	torch::Tensor distanceAccumulator = torch::zeros({classesCount}, torch::TensorOptions().dtype(torch::kFloat32));

	for(const auto& batch : loader)
	{
		torch::Tensor data = batch.data.to(*offload_device);
		torch::Tensor targets = batch.target.to(*offload_device);

		torch::Tensor output = network->forward(data);

		torch::Tensor loss = lossFn(output, targets);
		lossAccumulator += loss.template item<float>();

		distanceAccumulator += torch::pow(output-targets, 2*torch::ones({100, classesCount}, torch::TensorOptions().dtype(torch::kFloat32)));
	}

	std::cout<<"Parameter predictions:\n"<<torch::sqrt(distanceAccumulator);

	torch::Tensor distance = torch::sum(torch::sqrt(distanceAccumulator)/static_cast<double>(data_size));

	std::cout<<"Test Loss: "<<(lossAccumulator / data_size)*100<< "distance: " <<distance.template item<float>()<< std::endl;
	if(log)
		log->logTestLoss(epoch, data_size, (lossAccumulator / data_size)*100, distance.template item<float>(), data_size);
	return distance;
}

template <typename DatasetType, typename TestDatasetType = DatasetType>
int train(TrainLog* trainLog, std::shared_ptr<Net> net, EisDataset<DatasetType>* trainDataset, EisDataset<TestDatasetType>* testDataset,
		   size_t epochs, double learingRate)
{
	static constexpr double LOSS_START_DECADE = -2;
	static constexpr double LOSS_FINISH_DECADE = 6;
	static constexpr int64_t LOSS_POINT_COUNT = 3;

	assert(net->getOutputSize() == static_cast<int64_t>(trainDataset->classesCount()));
	assert(net->getInputSize() == trainDataset->get(0).data.size(0));
	assert(trainDataset->isMulticlass());
	assert(!net->hasSoftmaxOutput());
	if(testDataset)
	{
		assert(trainDataset->classesCount() == testDataset->classesCount());
		assert(testDataset->isMulticlass());
	}
	else
	{
		Log(Log::WARN)<<"No test dataset provided!";
	}

	std::string modelString = trainDataset->modelStringForClass(trainDataset->classesCount()+1);

	std::pair<torch::Tensor, torch::Tensor> scaleAndBias = getTargetScalesAndBiases(modelString);

	net->setOutputScalars(scaleAndBias.first.to(*offload_device));
	net->setOutputBiases(scaleAndBias.second.to(*offload_device));

	std::stringstream purpose;
	purpose<<"Regression,"<<modelString<<std::scientific;
	if(trainDataset->freqRange().has_value())
	{
		torch::Tensor freqs = trainDataset->freqRange().value();
		purpose<<','<<freqs[0].item().toDouble()<<','<<freqs[1].item().toDouble()<<','<<freqs[2].item().toDouble();
	}
	else
	{
		Log(Log::WARN)<<"Dataset dosent give a frequency range";
	}
	net->setPurpose(purpose.str());

	net->to(*offload_device);

	Log(Log::INFO)<<"Training "<<net->getPurpose();

	Log(Log::INFO)<<"Loss will simulate from 10^"<<LOSS_START_DECADE
		<<" to 10^"<<LOSS_FINISH_DECADE<<" with "<<LOSS_POINT_COUNT<<" steps "
		<<"and will be based on "<<modelString;
	torch::Tensor omega = torch::logspace(LOSS_START_DECADE, LOSS_FINISH_DECADE, LOSS_POINT_COUNT);
	torch::nn::MSELoss lossMse(torch::nn::MSELossOptions().reduction(torch::kMean));
	EisDistanceLoss lossEis(modelString, omega);

	std::vector<std::string> outputLables(net->getOutputSize());
	bool noLabels = false;
	for(int64_t i = 0; i < net->getOutputSize(); ++i)
	{
		std::string label = trainDataset->modelStringForClass(i);
		if(label == "Unkown")
		{
			Log(Log::WARN)<<"This dataset dosent provide output labels";
			noLabels = true;
			break;
		}
		outputLables[i] = trainDataset->modelStringForClass(i);
	}
	if(!noLabels)
		net->setOutputLabels(outputLables);

	torch::data::DataLoaderOptions options;
	options = options.batch_size(batch_size).workers(JOBS);
	auto trainDataLoader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>
		(trainDataset->map(torch::data::transforms::Stack<>()), options);
	auto testDataLoader =
		testDataset ? torch::data::make_data_loader(testDataset->map(torch::data::transforms::Stack<>()), options) : nullptr;

	torch::optim::AdamW optimizer(net->parameters(), torch::optim::AdamWOptions(learingRate).weight_decay(0));
	//torch::optim::SGD optimizer(net->parameters(), torch::optim::SGDOptions(learingRate).weight_decay(0));

	for (size_t epoch = 0; epoch < epochs; ++epoch)
	{
		int ret = trainImpl(net, *trainDataLoader,
			lossMse, optimizer, epoch, trainDataset->size().value(),
			trainDataset->classesCount(), trainLog);
		if(ret != 0)
			return -1;

		if(testDataset)
			test(net, *testDataLoader, lossMse, testDataset->size().value(), trainDataset->classesCount(), epoch, trainLog);

		trainLog->saveNetwork(net);
		Log(Log::INFO)<<"Epoch "<<epoch<<'/'<<epochs;
	}

	trainLog->saveNetwork(net);
	return 0;
}

}
}
