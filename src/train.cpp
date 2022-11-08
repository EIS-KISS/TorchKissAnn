#include <cstddef>
#include <cstdint>
#include <ios>
#include <iostream>
#include <eisgenerator/model.h>
#include <memory>
#include <vector>

#include "ann/ann.h"
#include "svm/svm.h"
#include "gan/gan.h"
#include "log.h"
#include "data/print.h"
#include "data/generators/eisgendata.h"
#include "data/generators/trashgenerator.h"
#include "data/loaders/dirloader.h"
#include "data/loaders/deduploader.h"
#include "trainoptions.h"

template <typename DataSetType>
int train(const Config& config);

template <typename DataSetType, typename TestDataSetType = DataSetType>
void trainSwitch(const TrainMode mode, EisDataset<DataSetType>* dataset, EisDataset<TestDataSetType>* testDataset);

template <> int train<FastDedupDataset>(const Config& config)
{
	FastDedupDataset dataset(config.fileName);

	if(!dataset.isOpen())
	{
		Log(Log::ERROR)<<"Could not load fast deduplicator dataset from "<<config.fileName;
		return 1;
	}

	trainSwitch<FastDedupDataset>(config.mode, &dataset, &dataset);
	return 0;
}

template <> int train<EisGeneratorDataset>(const Config& config)
{
	if(!config.fileName.empty() || !config.testFileName.empty())
		Log(Log::WARN)<<"warning: a file name was set, but thats meaningless with the generated dataset";
	EisGeneratorDataset* dataset =
		EisGeneratorDataset::getNeisGeneratedDataset(DATA_WIDTH, EisGeneratorDataset::DEFAULT_EXAMPLE_COUNT,
			config.noise, true);
	EisGeneratorDataset* testDataset =
		EisGeneratorDataset::getNeisGeneratedDataset(DATA_WIDTH, EisGeneratorDataset::DEFAULT_EXAMPLE_COUNT/11,
			config.noise, true);

	trainSwitch<EisGeneratorDataset>(config.mode, dataset, testDataset);
	return 0;
}

template <> int train<EisDirDataset>(const Config& config)
{
	if(config.fileName.empty() || std::filesystem::is_directory(config.fileName))
	{
		Log(Log::ERROR)<<"A valid directory must be specified in conbination with the dir dataset";
		return 1;
	}

	EisDirDataset dataset(config.fileName, "csv", FILE_TYPE_CSV);
	EisGeneratorDataset* testDataset =
		EisGeneratorDataset::getNeisGeneratedDataset(DATA_WIDTH, EisGeneratorDataset::DEFAULT_EXAMPLE_COUNT/11,
			config.noise, true);

	if(dataset.size().value() == 0)
	{
		Log(Log::ERROR)<<"Failed to load dataset from "<<config.fileName;
		return 2;
	}

	trainSwitch<EisDirDataset, EisGeneratorDataset>(config.mode, &dataset, testDataset);
	return 0;
}


int main(int argc, char** argv)
{
	std::cout<<std::setprecision(5)<<std::fixed<<std::setw(3);
	Log::level = Log::INFO;

	Config config;
	argp_parse(&argp, argc, argv, 0, 0, &config);

	svm_set_print_string_function(&svmPrintFunction);

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
void trainSwitch(const TrainMode mode, EisDataset<DataSetType>* dataset, EisDataset<TestDataSetType>* testDataset)
{
	Log(Log::INFO)<<"Training with"<<(dataset->isMulticlass() ? " muliclass" : "")<<
		" dataset of size "<<dataset->size().value()<<" with "<<dataset->classesCount()<<" classes";
	Log(Log::INFO)<<" Using weights:\n"<<dataset->classWeights()<<'\n';
	switch(mode)
	{
		case MODE_ANN:
			ann::train<DataSetType, TestDataSetType>("net.vos", dataset, testDataset, false);
			break;
		case MODE_ANN_CONV:
			ann::train<DataSetType, TestDataSetType>("netConv.vos", dataset, testDataset, true);
			break;
		case MODE_SVM:
			svm::train(false);
			break;
		case MODE_SVM_FILTER:
			svm::train(true);
			break;
		case MODE_SVM_CROSS:
			svm::cross();
			break;
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

