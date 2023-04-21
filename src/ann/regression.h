#pragma once
#include <cstddef>
#include <cstdint>
#include <torch/optim.h>
#include <string>

#include "net.h"
#include "constants.h"
#include "../data/generators/eisgendata.h"
#include "../utils/randomgen.h"
#include "../data/eistotorch.h"
#include "log.h"
#include "trainlog.h"
#include "tensoroperators.h"

namespace ann
{
namespace regression
{

torch::Tensor use(torch::Tensor input, std::shared_ptr<Net> net);

template <typename DatasetType>
void train(std::string fileName, EisDataset<DatasetType>* trainDataset, bool convNet, size_t epochs = 30);

template <typename DataLoader>
void trainImpl(std::shared_ptr<Net> network, DataLoader& loader, torch::optim::Optimizer& optimizer,
		   size_t epoch, size_t data_size, int64_t classesCount, TrainLog* log = nullptr)
{
	size_t index = 0;
	network->train();

	float lossAccumulator = 0;
	torch::Tensor distanceAccumulator = torch::zeros({classesCount}, torch::TensorOptions().dtype(torch::kFloat32));

	torch::nn::HuberLoss lossHuber(torch::nn::HuberLossOptions().delta(1));

	int64_t loginterval = data_size/10000;

	for(auto& batch : loader)
	{
		torch::Tensor data = batch.data.to(DATA_LOCATION);
		torch::Tensor targets = batch.target.to(DATA_LOCATION);

		torch::Tensor prediction = network->forward(data);
		torch::Tensor loss = lossHuber->forward(prediction, targets);

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
			auto end = std::min(data_size, (index + 1) * BATCH_SIZE);
			torch::Tensor distance = torch::sum(torch::sqrt(distanceAccumulator/static_cast<double>(end)));
			log->logTrainLoss(epoch, end, (lossAccumulator/end)*100, distance.template item<float>(), data_size);
		}
	}
}

template <typename DataLoader>
torch::Tensor test(std::shared_ptr<Net> network, DataLoader& loader, size_t data_size,
				   int64_t classesCount, size_t epoch = 0, TrainLog* log = nullptr)
{
	network->eval();
	torch::NoGradGuard noGrad;

	float lossAccumulator = 0;
	torch::Tensor distanceAccumulator = torch::zeros({classesCount}, torch::TensorOptions().dtype(torch::kFloat32));

	torch::nn::HuberLoss lossHuber(torch::nn::HuberLossOptions().delta(1));

	for(const auto& batch : loader)
	{
		torch::Tensor data = batch.data.to(DATA_LOCATION);
		torch::Tensor targets = batch.target.to(DATA_LOCATION);

		torch::Tensor output = network->forward(data);

		torch::Tensor loss = lossHuber->forward(output, targets);
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
	assert(net->getOutputSize() == static_cast<int64_t>(trainDataset->classesCount()));
	assert(trainDataset->isMulticlass());
	assert(!net->hasSoftmaxOutput());
	if(testDataset)
	{
		assert(trainDataset->classesCount() == testDataset->classesCount());
		assert(testDataset->isMulticlass());
	}

	Log(Log::WARN)<<"No test dataset provided!";

	net->setPurpose("Regression for " + trainDataset->modelStringForClass(trainDataset->classesCount()+1));

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
	options = options.batch_size(BATCH_SIZE).workers(JOBS);
	auto trainDataLoader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>
		(trainDataset->map(torch::data::transforms::Stack<>()), options);
	auto testDataLoader =
		testDataset ? torch::data::make_data_loader(testDataset->map(torch::data::transforms::Stack<>()), options) : nullptr;

	torch::optim::SGD optimizer(net->parameters(), torch::optim::SGDOptions(learingRate).momentum(0.5));

	for (size_t i = 0; i < epochs; ++i)
	{
		trainImpl(net, *trainDataLoader, optimizer, i, trainDataset->size().value(), trainDataset->classesCount(), trainLog);
		if(testDataset)
			test(net, *testDataLoader, testDataset->size().value(), trainDataset->classesCount(), i, trainLog);

		trainLog->saveNetwork(net);
		Log(Log::INFO)<<"Epoch "<<i<<'/'<<epochs;
	}

	trainLog->saveNetwork(net);
}

}
}
