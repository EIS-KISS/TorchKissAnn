#include "ann.h"
#include "log.h"

using namespace ann;

std::shared_ptr<Net> ann::load(const std::string& fileName, int64_t inputSize, int64_t outputSize, size_t downsampleSteps, size_t extraSteps)
{
	std::shared_ptr<SimpleNet> simpleNet(new SimpleNet(inputSize, outputSize, downsampleSteps, extraSteps));
	std::shared_ptr<ConvNet> convNet(new ConvNet(inputSize, outputSize, downsampleSteps, extraSteps));

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
		Log(Log::ERROR)<<fileName<<" is not a conv network or simple network with output size "<<outputSize
			<<" and "<<extraSteps<<" extra hidden layers";
		return nullptr;
	}
}

torch::Tensor ann::multiClassHits(const torch::Tensor& prediction, torch::Tensor targets, double thresh)
{
	torch::Tensor hits = torch::ones({prediction.size(0)},
		torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
	for(int64_t i = 0; i < prediction.size(0); ++i)
	{
		for(int64_t j = 0; j < prediction.size(1); ++j)
		{
			if(((targets[i][j].template item<float>()) >= thresh && prediction[i][j].template item<float>() < thresh) ||
				((targets[i][j].template item<float>()) < thresh && prediction[i][j].template item<float>() >= thresh))
			{
				hits[i] = 0;
			}
		}
	}

	return hits;
}

torch::Tensor ann::use(torch::Tensor input, std::shared_ptr<Net> net)
{
	net->eval();
	torch::Tensor output = torch::exp(net->forward(input.reshape({1, net->getInputSize()})));
	return output;
}

torch::Tensor ann::testOnGenNeis(std::shared_ptr<Net> net, std::string outputDir, double exportProb)
{
	EisGeneratorDataset* testDataset = EisGeneratorDataset::newDefaultDataset(DATA_WIDTH, 13e4);
	auto testDataLoader = torch::data::make_data_loader(testDataset->map(torch::data::transforms::Stack<>()),
														torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(16));

	net->eval();

	return test(net, *testDataLoader, testDataset->size().value(), testDataset->classesCount(),
				testDataset->classWeights(), false, outputDir, exportProb);
}
