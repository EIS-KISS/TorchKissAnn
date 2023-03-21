#include <cstddef>
#include <cstdint>
#include <ios>
#include <iostream>
#include <eisgenerator/model.h>
#include <memory>
#include <vector>

#include "ann/ann.h"
#include "gan/gan.h"
#include "log.h"
#include "data/print.h"
#include "data/splitdataset.h"
#include "data/generators/eisgendata.h"
#include "data/generators/trashgenerator.h"
#include "data/generators/extendordataset.h"
#include "data/loaders/dirloader.h"
#include "data/loaders/deduploader.h"
#include "options.h"
#include "trainlog.h"

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

int main(int argc, char** argv)
{
	std::cout<<std::setprecision(5)<<std::fixed<<std::setw(3);
	Log::level = Log::INFO;

	Config config;
	argp_parse(&argp, argc, argv, 0, 0, &config);

	if(config.mode == MODE_INVALID)
	{
		Log(Log::ERROR)<<"You must specify what to train: -m " MODE_LIST;
		return -1;
	}

	if(config.datasetMode == DATASET_INVALID)
	{
		Log(Log::ERROR)<<"You must specify what dataset to use: -d " DATASET_LIST;
		return -1;
	}

	TrainLog::initTrainLog();

	switch(config.datasetMode)
	{
		case DATASET_GEN:
			return train<EisGeneratorDataset>(config);
		case DATASET_FDUP:
			return train<FastDedupDataset>(config);
		case DATASET_DIR:
			return train<EisDirDataset>(config);
		case DATASET_TRASH:
			//return train<TrashDataset>(config);
		case DATASET_INVALID:
			Log(Log::ERROR)<<"You must specify a valid dataset to use: " DATASET_LIST;
			break;
	}

	return 0;
}

template <typename DataSetType, typename TestDataSetType>
void trainSwitch(const Config& config, EisDataset<DataSetType>* dataset, EisDataset<TestDataSetType>* testDataset)
{
	Log(Log::INFO)<<"Training with"<<(dataset->isMulticlass() ? " muliclass" : "")<<
		" dataset of size "<<dataset->size().value()<<" with "<<dataset->classesCount()<<" classes";
	Log(Log::INFO)<<" Using weights:\n"<<dataset->classWeights()<<'\n';

	{
		TrainLog::MetaData meta;
		meta.model = trainModeToStr(config.mode);
		meta.classNumber = dataset->classesCount();
		meta.multiClass = dataset->isMulticlass();
		meta.learingRate = config.learingRate;
		meta.extraLayers = config.extraLayers;
		meta.noise = config.noise;
		meta.dataset = datasetModeToStr(config.datasetMode);
		meta.datasetSize = dataset->size().value();
		meta.trainingFile = config.fileName;
		meta.testingFile = config.testFileName;
		saveMetadata(meta);
	}

	switch(config.mode)
	{
		case MODE_ANN:
		case MODE_ANN_CONV:
		{
			std::shared_ptr<ann::Net> net;
			std::string outputFileName("net");
			if(config.mode == MODE_ANN_CONV)
			{
				net.reset(new ann::ConvNet(DATA_WIDTH, dataset->classesCount(), 4, config.extraLayers));
				outputFileName.append("Conv");
			}
			else
			{
				net.reset(new ann::SimpleNet(DATA_WIDTH, dataset->classesCount(), 4, config.extraLayers));
			}
			outputFileName.append(std::to_string(dataset->classesCount()));
			outputFileName.push_back('_');
			outputFileName.append(std::to_string(config.extraLayers));
			outputFileName.append(".vos");
			ann::train<DataSetType, TestDataSetType>(outputFileName, net, dataset, testDataset, config.epochs, config.learingRate);
			break;
		}
		case MODE_GAN:
			gan::train("ganNet", EisGeneratorDataset::DEFAULT_EXAMPLE_COUNT);
			break;
		break;
		case MODE_INVALID:
		default:
			Log(Log::ERROR)<<"invalid model";
			break;
	}
}

