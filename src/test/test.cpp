#include <cstdint>
#include <ios>
#include <iostream>
#include <eisgenerator/model.h>
#include <eisgenerator/log.h>
#include <memory>
#include <vector>
#include <torch/torch.h>

#include "ann/classification.h"
#include "ann/regression.h"
#include "ann/scriptnet.h"
#include "ann/simplenet.h"
#include "ann/convnet.h"
#include "ann/simplenet.h"
#include "commonoptions.h"
#include "gan/gan.h"
#include "log.h"
#include "data/splitdataset.h"
#include "data/regressiondataset.h"
#include "data/generators/eisgendata.h"
#include "data/generators/eisgendatanoise.h"
#include "data/generators/parameterregressiondataset.h"
#include "data/loaders/regressionloader.h"
#include "data/loaders/dirloader.h"
#include "data/loaders/deduploader.h"
#include "data/loaders/coincellhellloader.h"
#include "data/passfaildataset.h"
#include "options.h"
#include "trainlog.h"

template <typename DataSetType>
int test(const Config& config);

template <typename DataSetType>
void testSwitch(const Config& config, DataSetType* testDataset);

template <>
int test<FastDedupDataset>(const Config& config)
{
	FastDedupDataset* dataset = new FastDedupDataset(config.fileName);

	if(!dataset->isOpen())
	{
		Log(Log::ERROR)<<"Could not load fast deduplicator dataset from "<<config.fileName;
		return 1;
	}

	testSwitch<FastDedupDataset>(config, dataset);
	delete dataset;
	return 0;
}

template <>
int test<EisGeneratorDataset>(const Config& config)
{
	EisGeneratorDataset* dataset;

	if(config.fileName.empty())
		dataset = EisGeneratorDataset::newDefaultDataset(config.dataWidth, config.datasetSize, config.noise, config.probeModel);
	else
		dataset = new EisGeneratorDataset(config.fileName, config.datasetSize, config.dataWidth, config.noise, config.probeModel);

	testSwitch<EisGeneratorDataset>(config, dataset);

	delete dataset;
	return 0;
}


template <>
int test<EisGeneratorDatasetNoise>(const Config& config)
{
	EisGeneratorDatasetNoise* dataset;
	EisGeneratorDatasetNoise* testDataset = nullptr;

	if(config.fileName.empty())
		dataset = EisGeneratorDatasetNoise::newDefaultDataset(config.dataWidth, config.datasetSize, config.probeModel);
	else
		dataset = new EisGeneratorDatasetNoise(config.fileName, config.datasetSize, config.dataWidth, config.probeModel);

	if(testDataset && dataset->outputSize() != testDataset->outputSize())
	{
		Log(Log::ERROR)<<"Training and test dataset dont have the same number of classes. "
		<<"The testg dataset has "<<dataset->outputSize()
		<<" and the test dataset has "<<testDataset->outputSize();
		return 1;
	}

	testSwitch<EisGeneratorDatasetNoise>(config, dataset);

	return 0;
}

template <>
int test<PassFaillDataset<EisGeneratorDataset>>(const Config& config)
{
	EisGeneratorDataset* origDataset = EisGeneratorDataset::newDefaultDataset(config.dataWidth, config.datasetSize, config.noise, config.probeModel);
	PassFaillDataset<EisGeneratorDataset> dataset(origDataset);

	testSwitch<PassFaillDataset<EisGeneratorDataset>>(config, &dataset);

	delete origDataset;
	return 0;
}

template <>
int test<ParameterRegressionDatasetGenerator>(const Config& config)
{
	if(config.fileName.empty())
	{
		Log(Log::ERROR)<<"Please specify a model as the filename for this dataset type";
		return 1;
	}

	ParameterRegressionDatasetGenerator dataset(config.fileName, config.dataWidth, 0, false, config.datasetSize);

	if(dataset.size().value() == 0)
	{
		Log(Log::ERROR)<<"Failed to create dataset for "<<config.fileName;
		return 2;
	}

	testSwitch<ParameterRegressionDatasetGenerator>(config, &dataset);
	return 0;
}

template <>
int test<ParameterRegressionDatasetLoader>(const Config& config)
{
	if(config.fileName.empty())
	{
		Log(Log::ERROR)<<"Please specify a filename of a valid tar file";
		return 1;
	}

	ParameterRegressionDatasetLoader dataset(config.fileName, true);

	testSwitch<ParameterRegressionDatasetLoader>(config, &dataset);
	return 0;
}

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

	switch(config.datasetMode)
	{
		case DATASET_GEN:
		case DATASET_GEN_NOISE:
		case DATASET_FDUP:
		case DATASET_PASSFAIL:
		case DATASET_DIR:
		{
			if(!net->getPurpose().starts_with("Classifier"))
			{
				Log(Log::ERROR)<<"Testing with this dataset only make sense with a classifier network";
				return;
			}
		}
		case DATASET_GEN_REGRESSION:
		case DATASET_TAR_REGRESSION:
		case DATASET_COINCELLHELL_POSTDICT:
		case DATASET_COINCELLHELL_PREDICT:
		{
			if(!net->getPurpose().starts_with("Classifier"))
			{
				Log(Log::ERROR)<<"Testing with this dataset only make sense with a classifier network";
				return;
			}
			torch::nn::MSELoss loss(torch::nn::MSELossOptions().reduction(torch::kMean));
			double result = ann::regression::test(net, *testDataLoader, loss, testDataset->size().value());
			Log(Log::INFO)<<"Result: "<<result;
		}
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

	if((config.datasetMode != DATASET_GEN_REGRESSION &&
		config.datasetMode != DATASET_TAR_REGRESSION &&
		config.datasetMode != DATASET_COINCELLHELL_PREDICT &&
		config.datasetMode != DATASET_COINCELLHELL_POSTDICT) &&
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
		case DATASET_GEN:
			return test<EisGeneratorDataset>(config);
		case DATASET_GEN_NOISE:
			return test<EisGeneratorDatasetNoise>(config);
		case DATASET_FDUP:
			return test<FastDedupDataset>(config);
		case DATASET_PASSFAIL:
			return test<PassFaillDataset<EisGeneratorDataset>>(config);
		case DATASET_DIR:
			return test<EisDirDataset>(config);
		case DATASET_GEN_REGRESSION:
			return test<ParameterRegressionDatasetGenerator>(config);
		case DATASET_TAR_REGRESSION:
			return test<ParameterRegressionDatasetLoader>(config);
		case DATASET_COINCELLHELL_POSTDICT:
			return test<CoinCellHellLoaderPostdict>(config);
		case DATASET_COINCELLHELL_PREDICT:
			return test<CoinCellHellLoaderPredict>(config);
		case DATASET_TAR:
		case DATASET_INVALID:
			Log(Log::ERROR)<<"You must specify a valid dataset to use: " DATASET_LIST;
			break;
	}

	free_device();
	return 0;
}
