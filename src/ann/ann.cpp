#include "ann.h"
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

torch::Tensor ann::testOnGenNeis(std::string fileName, std::string outputDir, double exportProb)
{
	EisGeneratorDataset* testDataset = EisGeneratorDataset::getNeisGeneratedDataset(DATA_WIDTH, EisGeneratorDataset::DEFAULT_SWEEP_DEPTH/2+2);
	auto testDataLoader = torch::data::make_data_loader(testDataset->map(torch::data::transforms::Stack<>()),
														torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(16));

	std::shared_ptr<Net> net(new Net);
	net->eval();
	torch::load(net, fileName);

	return test(net, *testDataLoader, testDataset->size().value(), testDataset->modelCount(), outputDir, exportProb);
}

void ann::train(std::string fileName, int sweepDepth, double noise)
{
	EisGeneratorDataset* trainDataset = EisGeneratorDataset::getNeisGeneratedDataset(DATA_WIDTH, sweepDepth, noise, true);
	EisGeneratorDataset* testDataset = EisGeneratorDataset::getNeisGeneratedDataset(DATA_WIDTH, sweepDepth/2+2, noise);

	std::shared_ptr<Net> net(new Net(DATA_WIDTH, trainDataset->modelCount()));

	LossLog log("annLoss.csv");

	torch::data::DataLoaderOptions options;
	options = options.batch_size(BATCH_SIZE).workers(JOBS);
	auto trainDataLoader = torch::data::make_data_loader(trainDataset->map(torch::data::transforms::Stack<>()), options);
	auto testDataLoader = torch::data::make_data_loader(testDataset->map(torch::data::transforms::Stack<>()), options);

	torch::optim::SGD optimizer(net->parameters(), torch::optim::SGDOptions(0.001).momentum(0.5));

	for (size_t i = 0; i < 3; ++i)
	{
		trainImpl(net, *trainDataLoader, optimizer, i+1, trainDataset->size().value(), trainDataset->modelWeights(), &log);
		test(net, *testDataLoader, testDataset->size().value(), trainDataset->modelCount());
		torch::save(net, fileName + ".uf");
	}

	delete trainDataset;
	delete testDataset;

	torch::save(net, fileName);
}
