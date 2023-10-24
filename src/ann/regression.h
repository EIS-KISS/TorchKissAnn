#pragma once
#include <ATen/ops/logspace.h>
#include <cstddef>
#include <cstdint>
#include <torch/optim.h>
#include <string>
#include <eisgenerator/model.h>
#include <torch/optim/adamw.h>

#include "net.h"
#include "globals.h"
#include "log.h"
#include "trainlog.h"
#include "eisdistanceloss.h"
#include "data/eisdataset.h"

namespace ann
{
namespace regression
{

torch::Tensor use(torch::Tensor input, std::shared_ptr<Net> net);

template <typename DatasetType>
void train(std::string fileName, EisDataset<DatasetType>* trainDataset, bool convNet, size_t epochs = 30);

template <typename DataLoader, typename LossFn>
void trainImpl(std::shared_ptr<Net> network, DataLoader& loader, LossFn& lossFn, torch::optim::Optimizer& optimizer,
		   size_t epoch, size_t data_size, int64_t classesCount, TrainLog* log = nullptr)
{
	size_t index = 0;
	network->train();

	float lossAccumulator = 0;
	torch::Tensor distanceAccumulator = torch::zeros({classesCount}, torch::TensorOptions().dtype(torch::kFloat32).device(*offload_device));

	int64_t loginterval = data_size/10000 ?: 1;

	for(auto& batch : loader)
	{
		torch::Tensor data = batch.data.to(*offload_device);
		torch::Tensor targets = batch.target.to(*offload_device);

		torch::Tensor prediction = network->forward(data);
		torch::Tensor loss = lossFn(prediction, targets);

		/*std::cout<<"data:\n"<<data<<'\n';
		std::cout<<"targets:\n"<<targets<<'\n';
		std::cout<<"prediction:\n"<<prediction<<'\n';
		std::cout<<"loss:\n"<<loss<<'\n';*/

		optimizer.zero_grad();
		loss.backward();
		optimizer.step();

		{
			torch::NoGradGuard noGrad;
			lossAccumulator += loss.template item<float>();
			torch::Tensor batchAccum = (prediction-targets).square().sum(0);
			distanceAccumulator += batchAccum;
		}

		if(log && index++ % loginterval == 0 && index / loginterval > 1)
		{
			auto end = std::min(data_size, (index + 1) * batch_size);
			torch::Tensor distance = torch::sum(torch::sqrt(distanceAccumulator/static_cast<double>(end)));
			distance = distance.to(torch::kCPU);
			log->logTrainLoss(epoch, end, (lossAccumulator/end)*100, distance.template item<float>(), data_size);
		}
	}
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
void train(TrainLog* trainLog, std::shared_ptr<Net> net, EisDataset<DatasetType>* trainDataset, EisDataset<TestDatasetType>* testDataset,
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

	net->setPurpose("Regression for " + modelString);
	net->to(*offload_device);

	Log(Log::INFO)<<"Training "<<net->getPurpose();

	Log(Log::INFO)<<"Loss will simulate from 10^"<<LOSS_START_DECADE
		<<" to 10^"<<LOSS_FINISH_DECADE<<" with "<<LOSS_POINT_COUNT<<" steps";
	torch::Tensor omega = torch::logspace(LOSS_START_DECADE, LOSS_FINISH_DECADE, LOSS_POINT_COUNT);
	torch::nn::BCEWithLogitsLoss lossBCE(torch::nn::BCEWithLogitsLossOptions().reduction(torch::kMean));
	EisDistanceLoss loss(modelString, omega);

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

	torch::optim::AdamW optimizer(net->parameters(), torch::optim::AdamWOptions(learingRate));

	for (size_t i = 0; i < epochs; ++i)
	{
		trainImpl(net, *trainDataLoader, loss, optimizer, i, trainDataset->size().value(), trainDataset->classesCount(), trainLog);
		if(testDataset)
			test(net, *testDataLoader, loss, testDataset->size().value(), trainDataset->classesCount(), i, trainLog);

		trainLog->saveNetwork(net);
		Log(Log::INFO)<<"Epoch "<<i<<'/'<<epochs;
	}

	trainLog->saveNetwork(net);
}

}
}
