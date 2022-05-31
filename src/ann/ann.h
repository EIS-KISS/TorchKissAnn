#pragma once
#include <torch/optim.h>
#include <string>

#include "net.h"
#include "constants.h"

namespace ann
{

torch::Tensor use(torch::Tensor input, std::string fileName);

void train(std::string fileName = "net.vos");

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


template <typename DataLoader>
void test(std::shared_ptr<Net> network, DataLoader& loader, size_t data_size)
{
	network->eval();
	torch::NoGradGuard no_grad;
	float Loss = 0, Acc = 0;

	for (const auto& batch : loader)
	{
		auto data = batch.data.to(DATA_LOCATION);
		auto targets = batch.target.to(DATA_LOCATION).view({-1});

		auto output = network->forward(data);
		auto loss = torch::nll_loss(output, targets);
		assert(!std::isnan(loss.template item<float>()));
		auto acc = output.argmax(1).eq(targets).sum();

		Loss += loss.template item<float>();
		Acc += acc.template item<float>();
	}

	std::cout<<"Test Loss: "<<Loss / data_size<< "\tAcc: " <<Acc / data_size<< std::endl;
}

}
