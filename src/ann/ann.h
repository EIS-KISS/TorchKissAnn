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
#include "losslog.h"
#include "tensoroperators.h"

namespace ann
{

std::shared_ptr<Net> load(const std::string& fileName = "net.vos");

torch::Tensor use(torch::Tensor input, std::shared_ptr<Net> net);

template <typename DatasetType>
void train(std::string fileName, EisDataset<DatasetType>* trainDataset, bool convNet, size_t epochs = 30);

torch::Tensor testOnGenNeis(std::shared_ptr<Net> net, std::string outputDir = "", double exportProb = 0);

torch::Tensor multiClassHits(const torch::Tensor& prediction, torch::Tensor targets, double thresh = 0.5);

template <typename DataLoader>
void trainImpl(std::shared_ptr<Net> network, DataLoader& loader, torch::optim::Optimizer& optimizer,
		   size_t epoch, size_t data_size, torch::Tensor classWeights, bool isMulticlass = false, LossLog* log = nullptr)
{
	size_t index = 0;
	network->train();
	float lossF = 0;
	float accF = 0;

	torch::nn::BCEWithLogitsLoss lossBCE(torch::nn::BCEWithLogitsLossOptions().reduction(torch::kMean).pos_weight(classWeights));
	torch::nn::NLLLoss lossNll(torch::nn::NLLLossOptions().reduction(torch::kMean).weight(classWeights));

	for(auto& batch : loader)
	{
		torch::Tensor data = batch.data.to(DATA_LOCATION);
		torch::Tensor targets = batch.target.to(DATA_LOCATION);

		torch::Tensor prediction = network->forward(data);
		torch::Tensor loss;

		torch::Tensor acc;
		if(!isMulticlass)
		{
			targets = targets.view({-1});
			loss = lossNll->forward(prediction, targets.to(torch::kInt64));
			acc = prediction.argmax(1).eq(targets).sum();
		}
		else
		{
			targets = targets.reshape({prediction.size(0), prediction.size(1)});

			loss = lossBCE->forward(prediction, targets.to(torch::kFloat32));
			acc = multiClassHits(torch::sigmoid(prediction), targets, 0.25).sum();
		}
		assert(!std::isnan(loss.template item<float>()));

		optimizer.zero_grad();
		loss.backward();
		optimizer.step();

		lossF += loss.template item<float>();
		accF += acc.template item<float>();

		if (index++ % 100 == 0 && index / 100 > 1)
		{
			auto end = std::min(data_size, (index + 1) * BATCH_SIZE);
			if(log)
				log->log(epoch, end, (lossF/end)*100, accF/end, data_size);
		}
	}
}

static const std::string getSampleOutputFileName(const std::string& dir, bool succ, int classNum)
{
	return dir + (succ ? std::string("/succ_") : std::string("/fail_")) +
			std::to_string(classNum) + std::string("_") + std::to_string(rd::uid()) + ".spc";
}

template <typename DataLoader>
torch::Tensor test(std::shared_ptr<Net> network, DataLoader& loader, size_t data_size,
				   int64_t classesCount, torch::Tensor classWeights, bool isMulticlass, const std::string& outputDir = "",
				   double outputProb = 0)
{
	network->eval();
	torch::NoGradGuard no_grad;
	float Loss = 0, Acc = 0;

	torch::Tensor classAcc = torch::zeros({classesCount}, torch::TensorOptions().dtype(torch::kInt64));
	torch::Tensor classCount = torch::zeros({classesCount}, torch::TensorOptions().dtype(torch::kInt64));

	torch::nn::BCEWithLogitsLoss lossBCE(torch::nn::BCEWithLogitsLossOptions().reduction(torch::kMean).weight(classWeights));
	torch::nn::NLLLoss lossNll(torch::nn::NLLLossOptions().reduction(torch::kMean).weight(classWeights));

	for(const auto& batch : loader)
	{
		torch::Tensor data = batch.data.to(DATA_LOCATION);
		torch::Tensor targets = batch.target.to(DATA_LOCATION).view({-1});

		torch::Tensor output = network->forward(data);
		torch::Tensor loss;

		if(!isMulticlass)
		{
			loss = lossNll->forward(output, targets.to(torch::kInt64));
			targets = unargmax(targets.to(torch::kInt64), classesCount);
			output = unargmax(output.argmax(1), classesCount);
		}
		else
		{
			targets = targets.reshape({output.size(0), output.size(1)});
			loss = lossBCE->forward(output, targets.to(torch::kFloat32));
			output = torch::sigmoid(output);
		}
		assert(!std::isnan(loss.template item<float>()));

		torch::Tensor hits = multiClassHits(output, targets, 0.25);

		for(int i = 0; i < hits.size(0); ++i)
		{
			if(hits[i].template item<int>() > 0)
			{
				classAcc += targets[i];
				if(outputProb > 0 && rd::rand() < (outputProb/10) && !outputDir.empty())
				{
					for(int64_t j = 0; j < targets[i].size(0); ++j)
					{
						if((targets[i][j].template item<int64_t>()) == 1)
							eis::saveToDisk(torchToEis(data.select(0, i)), getSampleOutputFileName(outputDir, true, j));
					}
				}
			}
			else if(outputProb > 0 && rd::rand() < outputProb && !outputDir.empty())
			{
				for(int64_t j = 0; j < targets[i].size(0); ++j)
				{
					if((targets[i][j].template item<int64_t>()) == 1)
						eis::saveToDisk(torchToEis(data.select(0, i)), getSampleOutputFileName(outputDir, false, j));
				}
			}
			classCount += targets[i];
		}
		torch::Tensor acc = hits.sum();

		Loss += loss.template item<float>();
		Acc += acc.template item<float>();
	}
	classAcc = classAcc/classCount;
	std::cout<<"Test class acc:\n"<<classAcc<<'\n';
	std::cout<<"Test Loss: "<<(Loss / data_size)*100<< "\tAcc: " <<Acc / data_size<< std::endl;
	return classAcc;
}

template <typename DatasetType, typename TestDatasetType = DatasetType>
void train(std::string fileName, EisDataset<DatasetType>* trainDataset, EisDataset<TestDatasetType>* testDataset, bool convNet, size_t epochs = 30)
{
	std::shared_ptr<Net> net;

	if(testDataset)
	{
		assert(trainDataset->classesCount() == testDataset->classesCount());
		assert(trainDataset->isMulticlass() == testDataset->isMulticlass());
	}

	if(convNet)
		net.reset(new ConvNet(DATA_WIDTH, trainDataset->classesCount()));
	else
		net.reset(new SimpleNet(DATA_WIDTH, trainDataset->classesCount()));

	LossLog log("annLoss.csv");

	torch::data::DataLoaderOptions options;
	options = options.batch_size(BATCH_SIZE).workers(JOBS);
	auto trainDataLoader = torch::data::make_data_loader(trainDataset->map(torch::data::transforms::Stack<>()), options);
	auto testDataLoader = torch::data::make_data_loader(testDataset->map(torch::data::transforms::Stack<>()), options);

	torch::optim::SGD optimizer(net->parameters(), torch::optim::SGDOptions(0.005).momentum(0.5));

	if(trainDataset->isMulticlass())
		Log(Log::DEBUG)<<"Using BCELoss";
	else
		Log(Log::DEBUG)<<"Using NLLLoss";

	for (size_t i = 0; i < epochs; ++i)
	{
		trainImpl(net, *trainDataLoader, optimizer, i+1, trainDataset->size().value(),
				  trainDataset->classWeights(), trainDataset->isMulticlass(), &log);
		if(testDataset)
		{
			test(net, *testDataLoader, testDataset->size().value(), trainDataset->classesCount(),
				 trainDataset->classWeights(), trainDataset->isMulticlass());
		}
		torch::save(net, fileName + ".uf");
		Log(Log::INFO)<<"Epoch "<<i<<'/'<<epochs;
	}

	torch::save(net, fileName);
}

}
