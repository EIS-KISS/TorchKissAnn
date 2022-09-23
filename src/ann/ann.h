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

namespace ann
{

std::shared_ptr<Net> load(const std::string& fileName = "net.vos");

torch::Tensor use(torch::Tensor input, std::shared_ptr<Net> net);

template <typename DatasetType>
void train(std::string fileName, EisDataset<DatasetType>* trainDataset, bool convNet);

torch::Tensor testOnGenNeis(std::shared_ptr<Net> net, std::string outputDir = "", double exportProb = 0);

template <typename DataLoader>
void trainImpl(std::shared_ptr<Net> network, DataLoader& loader, torch::optim::Optimizer& optimizer,
		   size_t epoch, size_t data_size, torch::Tensor classWeights, bool isMulticlass = false, LossLog* log = nullptr)
{
	size_t index = 0;
	network->train();
	float lossF = 0;
	float accF = 0;

	torch::nn::BCEWithLogitsLoss lossBCE(torch::nn::BCEWithLogitsLossOptions().reduction(torch::kMean).weight(classWeights));
	torch::nn::NLLLoss lossNll(torch::nn::NLLLossOptions().reduction(torch::kMean).weight(classWeights));

	for(auto& batch : loader)
	{
		torch::Tensor data = batch.data.to(DATA_LOCATION);
		torch::Tensor targets = batch.target.to(DATA_LOCATION).view({-1});

		torch::Tensor prediction = network->forward(data);
		torch::Tensor loss;

		torch::Tensor acc;
		if(!isMulticlass)
		{
			loss = lossNll->forward(prediction, targets.to(torch::kInt64));
			acc = prediction.argmax(1).eq(targets).sum();
		}
		else
		{
			targets = targets.reshape({prediction.size(0), prediction.size(1)});
			loss = lossBCE->forward(prediction, targets.to(torch::kFloat32));

			acc = torch::ones({prediction.size(0)},
				torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
			for(int64_t i = 0; i < prediction.size(0); ++i)
			{
				unsigned int hits = 0;
				acc[i] = 1;
				for(int64_t j = 0; j < prediction.size(1); ++j)
				{
					if((targets[i][j].template item<float>()) > 0.5)
					{
						++hits;
						if(prediction[i][j].template item<float>() < 0.5)
							acc[i] = 0;
					}
				}
			}
			acc = acc.sum();
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
				   int64_t classesCount, const std::string& outputDir = "", double outputProb = 0)
{
	network->eval();
	torch::NoGradGuard no_grad;
	float Loss = 0, Acc = 0;

	torch::Tensor classAcc = torch::zeros({classesCount}, torch::TensorOptions().dtype(torch::kInt64));
	torch::Tensor classCount = torch::zeros({classesCount}, torch::TensorOptions().dtype(torch::kInt64));

	for (const auto& batch : loader)
	{
		torch::Tensor data = batch.data.to(DATA_LOCATION);
		torch::Tensor targets = batch.target.to(DATA_LOCATION).view({-1});

		torch::Tensor output = network->forward(data);
		torch::Tensor loss = torch::nll_loss(output, targets.to(torch::kInt64));
		assert(!std::isnan(loss.template item<float>()));

		torch::Tensor hits = output.argmax(1).eq(targets);
		for(int i = 0; i < hits.size(0); ++i)
		{
			int classNum = targets[i].template item<int>();

			if(hits[i].template item<int>() > 0)
			{
				classAcc[classNum] += 1;
				if(outputProb > 0 && rd::rand() < (outputProb/10) && !outputDir.empty())
					eis::saveToDisk(torchToEis(data.select(0, i)), getSampleOutputFileName(outputDir, true, classNum));
			}
			else if(outputProb > 0 && rd::rand() < outputProb && !outputDir.empty())
			{
				eis::saveToDisk(torchToEis(data.select(0, i)), getSampleOutputFileName(outputDir, false, classNum));
			}
			classCount[classNum] += 1;
		}
		torch::Tensor acc = hits.sum();

		Loss += loss.template item<float>();
		Acc += acc.template item<float>();
	}
	classAcc = classAcc/classCount;
	std::cout<<"Test class acc:\n"<<classAcc<<'\n';
	std::cout<<"Test Loss: "<<Loss / data_size<< "\tAcc: " <<Acc / data_size<< std::endl;
	return classAcc;
}

template <typename DatasetType>
void train(std::string fileName, EisDataset<DatasetType>* trainDataset, bool convNet)
{
	EisDataset<EisGeneratorDataset>* testDataset = EisGeneratorDataset::getNeisGeneratedDataset(DATA_WIDTH, 13e4);

	std::shared_ptr<Net> net;

	if(convNet)
		net.reset(new ConvNet(DATA_WIDTH, trainDataset->classesCount()));
	else
		net.reset(new SimpleNet(DATA_WIDTH, trainDataset->classesCount()));

	LossLog log("annLoss.csv");

	torch::data::DataLoaderOptions options;
	options = options.batch_size(BATCH_SIZE).workers(JOBS);
	auto trainDataLoader = torch::data::make_data_loader(trainDataset->map(torch::data::transforms::Stack<>()), options);
	auto testDataLoader = torch::data::make_data_loader(testDataset->map(torch::data::transforms::Stack<>()), options);

	torch::optim::SGD optimizer(net->parameters(), torch::optim::SGDOptions(0.001).momentum(0.5));

	if(trainDataset->isMulticlass())
		Log(Log::DEBUG)<<"Using BCELoss";
	else
		Log(Log::DEBUG)<<"Using NLLLoss";

	for (size_t i = 0; i < 30; ++i)
	{
		trainImpl(net, *trainDataLoader, optimizer, i+1, trainDataset->size().value(),
				  trainDataset->classWeights(), trainDataset->isMulticlass(), &log);
		//test(net, *testDataLoader, testDataset->size().value(), trainDataset->classesCount());
		torch::save(net, fileName + ".uf");
	}

	delete testDataset;

	torch::save(net, fileName);
}

}
