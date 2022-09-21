#include "ann.h"
#include "log.h"

using namespace ann;

std::shared_ptr<Net> ann::load(const std::string& fileName)
{
	std::shared_ptr<SimpleNet> simpleNet(new SimpleNet);
	std::shared_ptr<ConvNet> convNet(new ConvNet);

	try
	{
		torch::load(simpleNet, fileName);
		Log(Log::DEBUG)<<fileName<<" is a simple network";
		return simpleNet;
	}
	catch(torch::Error& error)
	{
		Log(Log::INFO)<<fileName<<" is not a simple network";
	}

	try
	{
		torch::load(convNet, fileName);
		Log(Log::DEBUG)<<fileName<<" is a conv network";
		return convNet;
	}
	catch(torch::Error& error)
	{
		Log(Log::ERROR)<<fileName<<" is not a conv network or simple network";
		return nullptr;
	}
}

torch::Tensor ann::use(torch::Tensor input, std::shared_ptr<Net> net)
{
	net->eval();
	torch::Tensor output = torch::exp(net->forward(input.reshape({1, net->getInputSize()})));
	return output;
}

torch::Tensor ann::testOnGenNeis(std::shared_ptr<Net> net, std::string outputDir, double exportProb)
{
	EisGeneratorDataset* testDataset = EisGeneratorDataset::getNeisGeneratedDataset(DATA_WIDTH, 13e4);
	auto testDataLoader = torch::data::make_data_loader(testDataset->map(torch::data::transforms::Stack<>()),
														torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(16));

	net->eval();

	return test(net, *testDataLoader, testDataset->size().value(), testDataset->classesCount(), outputDir, exportProb);
}
