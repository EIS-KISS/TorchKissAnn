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

namespace ann
{

torch::Tensor use(torch::Tensor input, std::string fileName);

void train(std::string fileName = "net.vos", int sweepDepth = EisGeneratorDataset::DEFAULT_SWEEP_DEPTH);

torch::Tensor testOnGenNeis(std::string fileName = "net.vos", std::string outputDir = "", double exportProb = 0);

template <typename DataLoader>
void trainImpl(std::shared_ptr<Net> network, DataLoader& loader, torch::optim::Optimizer& optimizer,
		   size_t epoch, size_t data_size, torch::Tensor classWeights)
{
	std::cout<<__func__<<'\n';
	size_t index = 0;
	network->train();
	float Loss = 0, Acc = 0;

	for (auto& batch : loader)
	{
		auto data = batch.data.to(DATA_LOCATION);
		auto targets = batch.target.to(DATA_LOCATION).view({-1});

		auto prediction = network->forward(data);
		auto loss = torch::nll_loss(prediction, targets, classWeights);
		assert(!std::isnan(loss.template item<float>()));
		auto acc = prediction.argmax(1).eq(targets).sum();

		optimizer.zero_grad();
		loss.backward();
		optimizer.step();

		Loss += loss.template item<float>();
		Acc += acc.template item<float>();

		if (index++ % 1000 == 0)
		{
			auto end = std::min(data_size, (index + 1) * BATCH_SIZE);
			std::cout<<"Train Epoch: "<< epoch <<' '<<end<<'/'<< data_size<<"\tLoss: "<<Loss/end<<"\tAcc: "<<Acc/end<<'\n';
		}
	}
}

static const std::string getSampleOutputFileName(const std::string& dir, bool succ, int classNum)
{
	return dir + (succ ? std::string("/succ_") : std::string("/fail_")) +
			std::to_string(classNum) + std::string("_") + std::to_string(rd::uid());
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
		torch::Tensor loss = torch::nll_loss(output, targets);
		assert(!std::isnan(loss.template item<float>()));

		torch::Tensor hits = output.argmax(1).eq(targets);
		for(int i = 0; i < hits.size(0); ++i)
		{
			int classNum = targets[i].template item<int>();

			if(hits[i].template item<int>() > 0)
			{
				classAcc[classNum] += 1;
			}
			else if(outputProb > 0 && rd::rand() < outputProb && !outputDir.empty())
			{
				eis::saveToDisk(torchToEis(data.select(1, i)), getSampleOutputFileName(outputDir, false, classNum));
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

}
