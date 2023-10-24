#include <cstdint>
#include <ios>
#include <iostream>
#include <eisgenerator/model.h>
#include <memory>
#include <vector>
#include <torch/torch.h>

#include "ann/classification.h"
#include "ann/regression.h"
#include "data/loaders/regressionloader.h"
#include "gan/gan.h"
#include "log.h"
#include "data/splitdataset.h"
#include "data/generators/eisgendata.h"
#include "data/generators/extendordataset.h"
#include "data/generators/parameterregressiondataset.h"
#include "data/loaders/dirloader.h"
#include "data/loaders/deduploader.h"
#include "options.h"
#include "trainlog.h"
#include "globals.h"


template <typename DataSetType>
int train(const Config& config);

template <typename DataSetType>
void trainWithSplitDataset(const Config& config, DataSetType* dataset);

template <typename DataSetType, typename TestDataSetType = DataSetType>
void trainSwitch(const Config& config, EisDataset<DataSetType>* dataset, EisDataset<TestDataSetType>* testDataset);

template <typename DataSetType>
void trainWithSplitDataset(const Config& config, DataSetType* dataset)
{
	std::pair<SplitDataset<DataSetType>, SplitDataset<DataSetType>> datasets = createSplitDataset<DataSetType>(dataset, 0.1);
	trainSwitch<SplitDataset<DataSetType>, SplitDataset<DataSetType>>(config, &datasets.first, &datasets.second);
}

template <> int train<FastDedupDataset>(const Config& config)
{
	FastDedupDataset* dataset = new FastDedupDataset(config.fileName);

	if(!dataset->isOpen())
	{
		Log(Log::ERROR)<<"Could not load fast deduplicator dataset from "<<config.fileName;
		return 1;
	}

	if(!config.testFileName.empty())
	{
		Log(Log::WARN)<<"a seperate test dataset cant be used with EisDirDataset,"
			<<"instead 25% of the training dataset is automaticly used as a testing datastet";
	}

	trainWithSplitDataset<FastDedupDataset>(config, dataset);
	delete dataset;
	return 0;
}

template <> int train<EisGeneratorDataset>(const Config& config)
{
	EisGeneratorDataset* dataset;
	EisGeneratorDataset* testDataset = nullptr;

	if(config.fileName.empty())
		dataset = EisGeneratorDataset::newDefaultDataset(DATA_WIDTH, config.datasetSize, config.noise, config.probeModel);
	else
		dataset = new EisGeneratorDataset(config.fileName, config.datasetSize, DATA_WIDTH, config.noise, config.probeModel);

	if(config.testFileName.empty() && config.fileName.empty())
		testDataset = EisGeneratorDataset::newDefaultDataset(DATA_WIDTH, config.datasetSize/10, config.noise, config.probeModel);
	else if(!config.testFileName.empty())
		testDataset = new EisGeneratorDataset(config.testFileName, config.datasetSize/10, DATA_WIDTH, config.noise, config.probeModel, false);

	if(testDataset && dataset->classesCount() != testDataset->classesCount())
	{
		Log(Log::ERROR)<<"Training and test dataset dont have the same number of classes. "
			<<"The traing dataset has "<<dataset->classesCount()
			<<" and the test dataset has "<<testDataset->classesCount();
		return 1;
	}

	trainSwitch<EisGeneratorDataset>(config, dataset, testDataset);

	delete dataset;
	if(testDataset)
		delete testDataset;
	return 0;
}

template <> int train<EisDirDataset>(const Config& config)
{
	if(config.fileName.empty() || !std::filesystem::is_directory(config.fileName))
	{
		Log(Log::ERROR)<<"A valid directory must be specified in conbination with the dir dataset";
		return 1;
	}

	EisDirDataset dataset(config.fileName);
	EisDirDataset* testDataset = nullptr;

	if(dataset.size().value() == 0)
	{
		Log(Log::ERROR)<<"Failed to load dataset from "<<config.fileName;
		return 2;
	}

	if(!config.testFileName.empty())
	{
		testDataset = new EisDirDataset(config.testFileName);
		if(dataset.classesCount() != testDataset->classesCount())
		{
			Log(Log::ERROR)<<"Training and test dataset don't have the same number of classes. "
				<<"The traing dataset has "<<dataset.classesCount()
				<<" and the test dataset has "<<testDataset->classesCount();
			return 1;
		}
	}

	ExtendorDataset<EisDirDataset> datasetExtended(&dataset, 4);

	trainSwitch<ExtendorDataset<EisDirDataset>, EisDirDataset>(config, &datasetExtended, testDataset);

	if(testDataset)
		delete testDataset;
	return 0;
}

template <> int train<ParameterRegressionDatasetGenerator>(const Config& config)
{
	if(config.fileName.empty())
	{
		Log(Log::ERROR)<<"Please specify a model as the filename for this dataset type";
		return 1;
	}
	if(!config.testFileName.empty())
	{
		Log(Log::ERROR)<<"You can not specify a different model as the train dataset for this dataset type";
		return 1;
	}

	ParameterRegressionDatasetGenerator dataset(config.fileName, 100, 0, false, config.datasetSize);

	if(dataset.size().value() == 0)
	{
		Log(Log::ERROR)<<"Failed to create dataset for "<<config.fileName;
		return 2;
	}

	trainSwitch<ParameterRegressionDatasetGenerator, ParameterRegressionDatasetGenerator>(config, &dataset, nullptr);
	return 0;
}

template <> int train<ParameterRegressionDatasetLoader>(const Config& config)
{
	if(config.fileName.empty())
	{
		Log(Log::ERROR)<<"Please specify a filename of a valid tar file";
		return 1;
	}
	if(!config.testFileName.empty())
	{
		Log(Log::ERROR)<<"You can not specify a different tar archive as a testing dataset with this dataset type";
		return 1;
	}

	ParameterRegressionDatasetLoader dataset(config.fileName, true);

	if(dataset.size().value() == 0)
	{
		Log(Log::ERROR)<<"Failed to create dataset for "<<config.fileName;
		return 2;
	}

	trainSwitch<ParameterRegressionDatasetLoader, ParameterRegressionDatasetLoader>(config, &dataset, nullptr);
	return 0;
}

int main(int argc, char** argv)
{
	std::cout<<std::setprecision(5)<<std::fixed<<std::setw(3);
	Log::level = Log::INFO;

	Config config;
	argp_parse(&argp, argc, argv, 0, 0, &config);

	choose_device(config.noGpu);
	batch_size = config.batchSize;

	if(config.mode == MODE_INVALID)
	{
		Log(Log::ERROR)<<"You must specify what to train: -m " MODE_LIST;
		return 3;
	}

	if(config.datasetMode == DATASET_INVALID)
	{
		Log(Log::ERROR)<<"You must specify what dataset to use: -d " DATASET_LIST;
		return 3;
	}

	if((config.datasetMode != DATASET_GEN_REGRESSION && config.datasetMode != DATASET_TAR_REGRESSION) && config.mode == MODE_REGRESSION)
	{
		Log(Log::ERROR)<<"The regression mode can only be trained with a regression dataset";
		return 3;
	}

	TrainLog::setRunsDir("./runs");

	switch(config.datasetMode)
	{
		case DATASET_GEN:
			return train<EisGeneratorDataset>(config);
		case DATASET_FDUP:
			return train<FastDedupDataset>(config);
		case DATASET_DIR:
			return train<EisDirDataset>(config);
		case DATASET_GEN_REGRESSION:
			return train<ParameterRegressionDatasetGenerator>(config);
		case DATASET_TAR_REGRESSION:
			return train<ParameterRegressionDatasetLoader>(config);
		case DATASET_TRASH:
		case DATASET_TAR:
		case DATASET_INVALID:
			Log(Log::ERROR)<<"You must specify a valid dataset to use: " DATASET_LIST;
			break;
	}

	free_device();
	return 0;
}

template <typename DataSetType, typename TestDataSetType>
void trainSwitch(const Config& config, EisDataset<DataSetType>* trainDataset, EisDataset<TestDataSetType>* testDataset)
{
	Log(Log::INFO)<<"Training with"<<(trainDataset->isMulticlass() ? " muliclass" : "")<<
		" dataset of size "<<trainDataset->size().value()<<" with "<<trainDataset->classesCount()<<" classes";
	Log(Log::INFO)<<" Using weights:\n"<<trainDataset->classWeights()<<'\n';

	std::unique_ptr<TrainLog> trainLog;

	if(!config.outputDir.empty())
		trainLog.reset(new TrainLog(config.outputDir));
	else
		trainLog.reset(new TrainLog());

	{
		TrainLog::MetaData meta;
		meta.model = trainModeToStr(config.mode);
		meta.classNumber = trainDataset->classesCount();
		meta.multiClass = trainDataset->isMulticlass();
		meta.learingRate = config.learingRate;
		meta.extraLayers = config.extraLayers;
		meta.noise = config.noise;
		meta.dataset = datasetModeToStr(config.datasetMode);
		meta.datasetSize = trainDataset->size().value();
		meta.trainingFile = config.fileName;
		meta.testingFile = config.testFileName;
		trainLog->saveMetadata(meta);
	}

	switch(config.mode)
	{
		case MODE_ANN:
		case MODE_ANN_CONV:
		{
			Log(Log::INFO)<<"Training classification network";
			std::shared_ptr<ann::Net> net;
			if(config.mode == MODE_ANN_CONV)
				net.reset(new ann::ConvNet(DATA_WIDTH, trainDataset->classesCount(), 4, config.extraLayers, true));
			else
				net.reset(new ann::SimpleNet(DATA_WIDTH, trainDataset->classesCount(), 4, config.extraLayers, true));
			ann::classification::train<DataSetType, TestDataSetType>(trainLog.get(), net, trainDataset, testDataset, config.epochs, config.learingRate);
			break;
		}
		case MODE_GAN:
			gan::train("ganNet", EisGeneratorDataset::DEFAULT_EXAMPLE_COUNT);
			break;
		break;
		case MODE_REGRESSION:
		{
			Log(Log::INFO)<<"Training regression network";
			std::shared_ptr<ann::Net> net(new ann::SimpleNet(trainDataset->get(0).data.size(0),
															 trainDataset->classesCount(), 4, config.extraLayers, false));
			ann::regression::train(trainLog.get(), net, trainDataset, testDataset, config.epochs, config.learingRate);
			break;
		}
		case MODE_INVALID:
		default:
			Log(Log::ERROR)<<"invalid model";
			break;
	}
}
