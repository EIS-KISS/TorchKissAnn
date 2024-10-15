#include <iostream>
#include <eisgenerator/model.h>
#include <eisgenerator/log.h>
#include <memory>
#include <torch/torch.h>

#include "commonoptions.h"
#include "log.h"
#include "data/loaders/dirloader.h"
#include "options.h"
#include "globals.h"


template <typename DataSetType>
int test(const Config& config);

template <typename DataSetType>
void testSwitch(const Config& config, DataSetType* testDataset);

template <typename T>
int test(const Config& config)
{
	if(config.fileName.empty())
	{
		Log(Log::ERROR)<<"Please specify a filename of a valid tar file";
		return 1;
	}

	T dataset(config.fileName);

	if(dataset.size().value() == 0)
	{
		Log(Log::ERROR)<<"Failed to create dataset for "<<config.fileName;
		return 2;
	}

	testSwitch<T>(config, &dataset);
	return 0;
}

template <typename DataSetType>
void testSwitch(const Config& config, DataSetType* testDataset)
{
	Log(Log::INFO)<<"Testing with"<<(testDataset->isMulticlass() ? " muliclass" : "")<<
		" dataset of size "<<testDataset->size().value()<<" with "<<testDataset->outputSize()<<" classes";
	Log(Log::INFO)<<" Using weights:\n"<<testDataset->classWeights()<<'\n';

	std::shared_ptr<ann::Net> net(ann::Net::newNetFromCheckpointDir(config.netpath));

	torch::data::DataLoaderOptions options;
	options = options.batch_size(batch_size).workers(16);
	auto testDataLoader = torch::data::make_data_loader(testDataset->map(torch::data::transforms::Stack<>()), options);

	std::string purpose = net->getPurpose();

	switch(config.datasetMode)
	{
		default:
			Log(Log::ERROR)<<__func__<<" unsuported dataset";
			return;
	}
}

bool check_options(const Config& config)
{
	if(config.mode == MODE_INVALID)
	{
		Log(Log::ERROR)<<"You must specify what to test: -m " MODE_LIST;
		return false;
	}

	if(config.datasetMode == DATASET_INVALID)
	{
		Log(Log::ERROR)<<"You must specify what dataset to use: -d " DATASET_LIST;
		return false;
	}

	if((config.datasetMode != DATASET_DIR_REGRESSION &&
		config.datasetMode != DATASET_TAR_REGRESSION) &&
		(config.mode == MODE_REGRESSION || config.mode == MODE_REGRESSION_SCRIPT))
	{
		Log(Log::ERROR)<<"The regression mode can only be tested with a regression dataset";
		return false;
	}

	if(config.netpath.empty())
	{
		Log(Log::ERROR)<<"A network to test must be supplied via -n";
		return false;
	}

	return true;
}

int main(int argc, char** argv)
{
	std::cout<<std::setprecision(5)<<std::fixed<<std::setw(3);
	Log::level = Log::INFO;
	eis::Log::level = eis::Log::ERROR;

	Config config;
	argp_parse(&argp, argc, argv, 0, 0, &config);

	choose_device(config.noGpu);
	batch_size = config.batchSize;

	if(!check_options(config))
		return 3;

	Log(Log::INFO)<<"Testing "<<testModeToStr(config.mode);

	switch(config.datasetMode)
	{
		case DATASET_DIR:
			return test<EisDirDataset>(config);
		case DATASET_DIR_REGRESSION:
		case DATASET_TAR_REGRESSION:
		case DATASET_TAR:
		case DATASET_INVALID:
			Log(Log::ERROR)<<"You must specify a valid dataset to use: " DATASET_LIST;
			break;
	}

	free_device();
	return 0;
}
