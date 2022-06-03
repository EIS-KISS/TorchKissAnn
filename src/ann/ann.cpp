#include "ann.h"
#include "../data/generators/eisgendata.h"
#include "log.h"

using namespace ann;

torch::Tensor ann::use(torch::Tensor input, std::string fileName)
{
	std::shared_ptr<Net> net(new Net);
	net->eval();
	torch::load(net, fileName);
	torch::Tensor output = torch::exp(net->forward(input.reshape({1, net->getInputSize()})));
	return output;
}

torch::Tensor ann::testOnGenNeis(std::string fileName)
{
	EisGeneratorDataset* testDataset = EisGeneratorDataset::getNeisGeneratedDatasetAlt(DATA_WIDTH);
	auto testDataLoader = torch::data::make_data_loader(testDataset->map(torch::data::transforms::Stack<>()),
														torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(16));

	std::shared_ptr<Net> net(new Net);
	net->eval();
	torch::load(net, fileName);

	return test(net, *testDataLoader, testDataset->size().value(), testDataset->modelCount());
}

void ann::train(std::string fileName)
{
	EisGeneratorDataset* trainDataset = EisGeneratorDataset::getNeisGeneratedDatasetWithBLl(DATA_WIDTH);
	EisGeneratorDataset* testDataset = EisGeneratorDataset::getNeisGeneratedDatasetAlt(DATA_WIDTH);

	std::shared_ptr<Net> net(new Net(DATA_WIDTH, trainDataset->modelCount()));

	torch::data::DataLoaderOptions options;
	options = options.batch_size(BATCH_SIZE).workers(16);
	auto trainDataLoader = torch::data::make_data_loader(trainDataset->map(torch::data::transforms::Stack<>()), options);
	auto testDataLoader = torch::data::make_data_loader(testDataset->map(torch::data::transforms::Stack<>()), options);

	torch::optim::SGD optimizer(net->parameters(), torch::optim::SGDOptions(0.001).momentum(0.5));

	for (size_t i = 0; i < 30; ++i)
	{
		trainImpl(net, *trainDataLoader, optimizer, i+1, trainDataset->size().value(), trainDataset->modelWeights());
		test(net, *testDataLoader, testDataset->size().value(), trainDataset->modelCount());
		torch::save(net, fileName + ".uf");
	}

	delete trainDataset;
	delete testDataset;

	torch::save(net, fileName);
}
