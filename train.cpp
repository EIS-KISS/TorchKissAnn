#include <iostream>
#include <eisgenerator/model.h>
#include <memory>
#include <torch/optim.h>

#include "net.h"
#include "eisgendata.h"
#include "testgenerator.h"
#include "log.h"

static constexpr int batch_size = 100;
static constexpr auto DataLocation = torch::kCPU;

template <typename DataLoader>
void train(std::shared_ptr<Net> network, DataLoader& loader, torch::optim::Optimizer& optimizer, size_t epoch, size_t data_size)
{
	std::cout<<__func__<<'\n';
	size_t index = 0;
	network->train();
	float Loss = 0, Acc = 0;

	for (auto& batch : loader)
	{
		auto data = batch.data.to(DataLocation);
		auto targets = batch.target.to(DataLocation).view({-1});

		auto prediction = network->forward(data);
		auto loss = torch::nll_loss(prediction, targets);
		assert(!std::isnan(loss.template item<float>()));
		auto acc = prediction.argmax(1).eq(targets).sum();

		optimizer.zero_grad();
		loss.backward();
		optimizer.step();

		Loss += loss.template item<float>();
		Acc += acc.template item<float>();

		if (index++ % 1000 == 0)
		{
			auto end = std::min(data_size, (index + 1) * batch_size);
			std::cout<<"Train Epoch: "<< epoch <<' '<<end<<'/'<< data_size<<"\tLoss: "<<Loss/end<<"\tAcc: "<<Acc/end<<'\n';
			torch::save(network, "net.pt");
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
		auto data = batch.data.to(DataLocation);
		auto targets = batch.target.to(DataLocation).view({-1});

		auto output = network->forward(data);
		auto loss = torch::nll_loss(output, targets);
		assert(!std::isnan(loss.template item<float>()));
		auto acc = output.argmax(1).eq(targets).sum();

		Loss += loss.template item<float>();
		Acc += acc.template item<float>();
	}

	std::cout<<"Test Loss: "<<Loss / data_size<< "\tAcc: " <<Acc / data_size<< std::endl;
}

int main(int argc, char** argv)
{
	Log::level = Log::DEBUG;
	std::shared_ptr<Net> net(new Net);

	EisGeneratorDataset* trainDataset = EisGeneratorDataset::getNeisGeneratedDatasetWithBLl();
	EisGeneratorDataset* testDataset = EisGeneratorDataset::getNeisGeneratedDatasetAlt();

	torch::data::DataLoaderOptions options;
	options = options.batch_size(batch_size).workers(16);
	auto trainDataLoader = torch::data::make_data_loader(trainDataset->map(torch::data::transforms::Stack<>()), options);
	auto testDataLoader = torch::data::make_data_loader(testDataset->map(torch::data::transforms::Stack<>()), options);

	torch::optim::SGD optimizer(net->parameters(), torch::optim::SGDOptions(0.001).momentum(0.5));

	for (size_t i = 0; i < 4; ++i)
	{
		train(net, *trainDataLoader, optimizer, i+1, trainDataset->size().value());
		test(net, *testDataLoader, testDataset->size().value());
	}

	delete trainDataset;
	delete testDataset;

	torch::save(net, "net.pt");

	return 0;
}
