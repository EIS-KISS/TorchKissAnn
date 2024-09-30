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
#include "data/splitregressiondataset.h"
#include "data/passfaildataset.h"
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
		dataset = EisGeneratorDataset::newDefaultDataset(config.dataWidth, config.datasetSize, config.noise, config.probeModel);
	else
		dataset = new EisGeneratorDataset(config.fileName, config.datasetSize, config.dataWidth, config.noise, config.probeModel);

	if(config.testFileName.empty() && config.fileName.empty())
		testDataset = EisGeneratorDataset::newDefaultDataset(config.dataWidth, config.datasetSize/10, config.noise, config.probeModel);
	else if(!config.testFileName.empty())
		testDataset = new EisGeneratorDataset(config.testFileName, config.datasetSize/10, config.dataWidth, config.noise, config.probeModel, false);

	if(testDataset && dataset->outputSize() != testDataset->outputSize())
	{
		Log(Log::ERROR)<<"Training and test dataset dont have the same number of classes. "
			<<"The traing dataset has "<<dataset->outputSize()
			<<" and the test dataset has "<<testDataset->outputSize();
		return 1;
	}

	trainSwitch<EisGeneratorDataset>(config, dataset, testDataset);

	delete dataset;
	if(testDataset)
		delete testDataset;
	return 0;
}


template <> int train<EisGeneratorDatasetNoise>(const Config& config)
{
	EisGeneratorDatasetNoise* dataset;
	EisGeneratorDatasetNoise* testDataset = nullptr;

	if(config.fileName.empty())
		dataset = EisGeneratorDatasetNoise::newDefaultDataset(config.dataWidth, config.datasetSize, config.probeModel);
	else
		dataset = new EisGeneratorDatasetNoise(config.fileName, config.datasetSize, config.dataWidth, config.probeModel);

	if(config.testFileName.empty() && config.fileName.empty())
		testDataset = EisGeneratorDatasetNoise::newDefaultDataset(config.dataWidth, config.datasetSize/10, config.probeModel);
	else if(!config.testFileName.empty())
		testDataset = new EisGeneratorDatasetNoise(config.testFileName, config.datasetSize/10, config.dataWidth, config.probeModel);

	if(testDataset && dataset->outputSize() != testDataset->outputSize())
	{
		Log(Log::ERROR)<<"Training and test dataset dont have the same number of classes. "
		<<"The traing dataset has "<<dataset->outputSize()
		<<" and the test dataset has "<<testDataset->outputSize();
		return 1;
	}

	trainSwitch<EisGeneratorDatasetNoise>(config, dataset, testDataset);

	delete dataset;
	if(testDataset)
		delete testDataset;
	return 0;
}

template <> int train<PassFaillDataset<EisGeneratorDataset>>(const Config& config)
{
	EisGeneratorDataset* origDataset = EisGeneratorDataset::newDefaultDataset(config.dataWidth, config.datasetSize, config.noise, config.probeModel);
	PassFaillDataset<EisGeneratorDataset> dataset(origDataset);

	EisGeneratorDataset* origTestDataset = EisGeneratorDataset::newDefaultDataset(config.dataWidth, 3000, config.noise, config.probeModel);
	PassFaillDataset<EisGeneratorDataset> testDataset(origTestDataset);

	trainSwitch<PassFaillDataset<EisGeneratorDataset>, PassFaillDataset<EisGeneratorDataset>>(config, &dataset, &testDataset);

	delete origDataset;
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
		if(dataset.outputSize() != testDataset->outputSize())
		{
			Log(Log::ERROR)<<"Training and test dataset don't have the same number of classes. "
				<<"The traing dataset has "<<dataset.outputSize()
				<<" and the test dataset has "<<testDataset->outputSize();
			return 1;
		}
	}

	trainSwitch<EisDirDataset, EisDirDataset>(config, &dataset, testDataset);

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

	ParameterRegressionDatasetGenerator dataset(config.fileName, config.dataWidth, 0, false, config.datasetSize);

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

template <> int train<CoinCellHellLoaderPostdict>(const Config& config)
{
	if(config.fileName.empty())
	{
		Log(Log::ERROR)<<"Please specify a filename of a valid tar file";
		return 1;
	}
	if(!config.testFileName.empty())
	{
		Log(Log::ERROR)<<"You can not specify a different tar archive as a testing dataset with this dataset type, the train datase will be automaticly split";
		return 1;
	}

	CoinCellHellLoaderPostdict dataset(config.fileName);
	std::pair<SplitRegressionDataset<CoinCellHellLoaderPostdict>, SplitRegressionDataset<CoinCellHellLoaderPostdict>> splitdatasets = SplitRegressionDataset<CoinCellHellLoaderPostdict>::createSplitRegressionDataset(&dataset, 0.1);

	if(dataset.size().value() == 0)
	{
		Log(Log::ERROR)<<"Failed to create dataset for "<<config.fileName;
		return 2;
	}

	trainSwitch<SplitRegressionDataset<CoinCellHellLoaderPostdict>, SplitRegressionDataset<CoinCellHellLoaderPostdict>>(config, &splitdatasets.first, &splitdatasets.second);
	return 0;
}

template <> int train<CoinCellHellLoaderPredict>(const Config& config)
{
	if(config.fileName.empty())
	{
		Log(Log::ERROR)<<"Please specify a filename of a valid tar file";
		return 1;
	}
	if(!config.testFileName.empty())
	{
		Log(Log::ERROR)<<"You can not specify a different tar archive as a testing dataset with this dataset type, the train datase will be automaticly split";
		return 1;
	}

	CoinCellHellLoaderPredict dataset(config.fileName);
	std::pair<SplitRegressionDataset<CoinCellHellLoaderPredict>, SplitRegressionDataset<CoinCellHellLoaderPredict>> splitdatasets = SplitRegressionDataset<CoinCellHellLoaderPredict>::createSplitRegressionDataset(&dataset, 0.1);

	if(dataset.size().value() == 0)
	{
		Log(Log::ERROR)<<"Failed to create dataset for "<<config.fileName;
		return 2;
	}

	trainSwitch<SplitRegressionDataset<CoinCellHellLoaderPredict>, SplitRegressionDataset<CoinCellHellLoaderPredict>>(config, &splitdatasets.first, &splitdatasets.second);
	return 0;
}

template <typename DataSetType, typename TestDataSetType>
void trainSwitch(const Config& config, EisDataset<DataSetType>* trainDataset, EisDataset<TestDataSetType>* testDataset)
{
	Log(Log::INFO)<<"Training with"<<(trainDataset->isMulticlass() ? " muliclass" : "")<<
		" dataset of size "<<trainDataset->size().value()<<" with "<<trainDataset->outputSize()<<" classes";
	Log(Log::INFO)<<" Using weights:\n"<<trainDataset->classWeights()<<'\n';

	std::unique_ptr<TrainLog> trainLog;

	if(!config.outputDir.empty())
		trainLog.reset(new TrainLog(config.outputDir));
	else
		trainLog.reset(new TrainLog());

	{
		TrainLog::MetaData meta;
		meta.model = trainModeToStr(config.mode);
		meta.classNumber = trainDataset->outputSize();
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
		{
			Log(Log::INFO)<<"Training classification network with simple net";
			std::shared_ptr<ann::Net> net(new ann::SimpleNet(trainDataset->inputSize(), trainDataset->outputSize(), 4, config.extraLayers, true));
			ann::classification::train<DataSetType, TestDataSetType>(trainLog.get(), net, trainDataset, testDataset, config.epochs, config.learingRate);
			break;
		}
		case MODE_ANN_CONV:
		{
			Log(Log::INFO)<<"Training classification network with conv net";
			std::shared_ptr<ann::Net> net(new ann::ConvNet(trainDataset->inputSize(), trainDataset->outputSize(), 4, config.extraLayers, true));
			ann::classification::train<DataSetType, TestDataSetType>(trainLog.get(), net, trainDataset, testDataset, config.epochs, config.learingRate);
			break;
		}
		case MODE_ANN_SCRIPT:
		{
			try
			{
				Log(Log::INFO)<<"Training classification network with script net loaded from "<<config.scriptPath;
				std::shared_ptr<ann::Net> net(new ann::ScriptNet(config.scriptPath, true, config.dataWidth, trainDataset->outputSize()));
				ann::classification::train<DataSetType, TestDataSetType>(trainLog.get(), net, trainDataset, testDataset, config.epochs, config.learingRate);
			}
			catch(const ann::ScriptNet::load_errror& err)
			{
				Log(Log::ERROR)<<"Could not load TorchScript model: "<<err.what();
			}
			break;
		}
		case MODE_GAN:
		{
			EisGeneratorDataset dataset;
			gan::train(trainLog.get(), &dataset, config.epochs, config.learingRate);
			break;
		}
		break;
		case MODE_REGRESSION:
		{
			RegressionDataset<DataSetType>* trainDatasetReg = dynamic_cast<RegressionDataset<DataSetType>*>(trainDataset);
			RegressionDataset<DataSetType>* testDatasetReg = dynamic_cast<RegressionDataset<DataSetType>*>(testDataset);
			if(!trainDatasetReg || (testDataset && !testDatasetReg))
			{
				Log(Log::ERROR)<<"can not use a dataset of type "<<typeid(trainDataset).name()<<" to train regression as this is not a regression dataset";
				exit(1);
			}
			Log(Log::INFO)<<"Training regression network "<<trainDataset->inputSize()<<' '<<trainDataset->outputSize();
			std::shared_ptr<ann::Net> net(new ann::SimpleNet(trainDataset->inputSize(),
															 trainDataset->outputSize(), 4, config.extraLayers, false));
			ann::regression::train(trainLog.get(), net, trainDatasetReg, testDatasetReg, config.epochs, config.learingRate);
			break;
		}
		case MODE_REGRESSION_SCRIPT:
		{
			RegressionDataset<DataSetType>* trainDatasetReg = dynamic_cast<RegressionDataset<DataSetType>*>(trainDataset);
			RegressionDataset<DataSetType>* testDatasetReg = dynamic_cast<RegressionDataset<DataSetType>*>(testDataset);
			if(!trainDatasetReg || (testDataset && !testDatasetReg))
			{
				Log(Log::ERROR)<<"can not use a dataset of type "<<typeid(trainDataset).name()<<" to train regression as this is not a regression dataset";
				exit(1);
			}
			Log(Log::INFO)<<"Training regression script network "<<trainDataset->inputSize()<<' '<<trainDataset->outputSize();
			std::shared_ptr<ann::Net> net(new ann::ScriptNet(config.scriptPath, false, trainDataset->inputSize(), trainDataset->outputSize()));
			ann::regression::train(trainLog.get(), net, trainDatasetReg, testDatasetReg, config.epochs, config.learingRate);
			break;
		}
		case MODE_INVALID:
		default:
			Log(Log::ERROR)<<"invalid model";
			break;
	}
}

bool check_options(const Config& config)
{
	if(config.mode == MODE_INVALID)
	{
		Log(Log::ERROR)<<"You must specify what to train: -m " MODE_LIST;
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
		Log(Log::ERROR)<<"The regression mode can only be trained with a regression dataset";
		return false;
	}

	if((config.mode == MODE_ANN_SCRIPT || config.mode == MODE_REGRESSION_SCRIPT) && config.scriptPath.empty())
	{
		Log(Log::ERROR)<<"To train a TorchScript a path to a TorchScript must be supplied via -n";
		return false;
	}

	if((config.mode != MODE_ANN_SCRIPT && config.mode != MODE_REGRESSION_SCRIPT) && !config.scriptPath.empty())
	{
		Log(Log::ERROR)<<"a path to a TorchScript can only be supplied in "<<trainModeToStr(MODE_ANN_SCRIPT)<<" mode";
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

	Log(Log::INFO)<<"Training "<<trainModeToStr(config.mode);

	TrainLog::setRunsDir("./runs");

	switch(config.datasetMode)
	{
		case DATASET_GEN:
			return train<EisGeneratorDataset>(config);
		case DATASET_GEN_NOISE:
			return train<EisGeneratorDatasetNoise>(config);
		case DATASET_FDUP:
			return train<FastDedupDataset>(config);
		case DATASET_PASSFAIL:
			return train<PassFaillDataset<EisGeneratorDataset>>(config);
		case DATASET_DIR:
			return train<EisDirDataset>(config);
		case DATASET_GEN_REGRESSION:
			return train<ParameterRegressionDatasetGenerator>(config);
		case DATASET_TAR_REGRESSION:
			return train<ParameterRegressionDatasetLoader>(config);
		case DATASET_COINCELLHELL_POSTDICT:
			return train<CoinCellHellLoaderPostdict>(config);
		case DATASET_COINCELLHELL_PREDICT:
			return train<CoinCellHellLoaderPredict>(config);
		case DATASET_INVALID:
			Log(Log::ERROR)<<"You must specify a valid dataset to use: " DATASET_LIST;
			break;
		default:
			Log(Log::ERROR)<<"Dataset not implemented";
			break;
	}

	free_device();
	return 0;
}
