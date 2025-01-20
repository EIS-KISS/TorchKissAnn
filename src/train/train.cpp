#include <cerrno>
#include <cstdint>
#include <filesystem>
#include <ios>
#include <iostream>
#include <eisgenerator/model.h>
#include <eisgenerator/log.h>
#include <kisstype/spectra.h>
#include <kisstype/type.h>
#include <memory>
#include <string>
#include <torch/torch.h>

#include "ann/autoencode.h"
#include "ann/autoencoder.h"
#include "ann/classification.h"
#include "ann/regression.h"
#include "ann/scriptnet.h"
#include "ann/simplenet.h"
#include "ann/convnet.h"
#include "commonoptions.h"
#include "data/loaders/regressiondirloader.h"
#include "data/loaders/tarloader.h"
#include "log.h"
#include "data/regressiondataset.h"
#include "data/loaders/regressionloader.h"
#include "data/loaders/dirloader.h"
#include "options.h"
#include "trainlog.h"
#include "tokenize.h"

template <typename DataSetType>
int train(const Config& config);

template <typename DataSetType>
void trainWithSplitDataset(const Config& config, DataSetType* dataset);

template <typename DataSetType, typename TestDataSetType = DataSetType>
void trainSwitch(const Config& config, EisDataset<DataSetType>* dataset, EisDataset<TestDataSetType>* testDataset);

template <typename DataSetType>
int train(const Config& config)
{
	if(config.fileName.empty())
	{
		Log(Log::ERROR)<<"A valid file or directoy must be specified in conbination with the "<<datasetModeToStr(config.datasetMode)<<" dataset";
		return 1;
	}

	DataSetType dataset(config.fileName);
	DataSetType* testDataset = nullptr;

	if(dataset.size().value() == 0)
	{
		Log(Log::ERROR)<<"Failed to load dataset from "<<config.fileName;
		return 2;
	}

	if(!config.testFileName.empty())
	{
		testDataset = new DataSetType(config.testFileName);
		if(dataset.outputSize() != testDataset->outputSize())
		{
			Log(Log::ERROR)<<"Training and test dataset don't have the same number of classes. "
				<<"The traing dataset has "<<dataset.outputSize()
				<<" and the test dataset has "<<testDataset->outputSize();
			return 1;
		}
	}

	trainSwitch<DataSetType, DataSetType>(config, &dataset, testDataset);

	if(testDataset)
		delete testDataset;
	return 0;
}

template <typename DataSetType, typename TestDataSetType>
void trainSwitch(const Config& config, EisDataset<DataSetType>* trainDataset, EisDataset<TestDataSetType>* testDataset)
{
	Log(Log::INFO)<<"Training with"<<(trainDataset->isMulticlass() ? " muliclass" : "")<<
		" dataset of size "<<trainDataset->size().value()<<" with "<<trainDataset->outputSize()<<" classes";
	if(!config.noWeights)
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
			ann::classification::train<DataSetType, TestDataSetType>(
				trainLog.get(),
				net,
				trainDataset,
				testDataset,
				config.epochs,
				config.learingRate,
				config.noWeights);
			break;
		}
		case MODE_ANN_CONV:
		{
			Log(Log::INFO)<<"Training classification network with conv net";
			std::shared_ptr<ann::Net> net(new ann::ConvNet(trainDataset->inputSize(), trainDataset->outputSize(), 4, config.extraLayers, true));
			ann::classification::train<DataSetType, TestDataSetType>(
				trainLog.get(),
				net,
				trainDataset,
				testDataset,
				config.epochs,
				config.learingRate,
				config.noWeights);
			break;
		}
		case MODE_ANN_SCRIPT:
		{
			try
			{
				Log(Log::INFO)<<"Training classification network with script net loaded from "<<config.scriptPath;
				std::shared_ptr<ann::Net> net(new ann::ScriptNet(config.scriptPath, true, trainDataset->inputSize(), trainDataset->outputSize()));
				ann::classification::train<DataSetType, TestDataSetType>(
					trainLog.get(),
					net,
					trainDataset,
					testDataset,
					config.epochs,
					config.learingRate,
					config.noWeights);
			}
			catch(const ann::ScriptNet::load_errror& err)
			{
				Log(Log::ERROR)<<"Could not load TorchScript model: "<<err.what();
			}
			break;
		}
		case MODE_AUTO_SCRIPT:
		{
			try
			{
				std::vector<std::string> tokens = tokenize(config.scriptPath, ',');
				if(tokens.size() != 2)
				{
					Log(Log::ERROR)<<"For Autoencoders the network path given by -n must be two paths seperated by a: /path/a,/path/b";
					return;
				}
				Log(Log::INFO)<<"Training autoencoder network with script net loaded from "<<config.scriptPath;
				std::shared_ptr<ann::Net> encoder(new ann::ScriptNet(tokens[0], false, trainDataset->inputSize(), config.latentSize));
				std::shared_ptr<ann::Net> decoder(new ann::ScriptNet(tokens[1], false, config.latentSize, trainDataset->inputSize()));
				std::shared_ptr<ann::AutoEncoder> autoencoder(new ann::AutoEncoder(encoder, decoder));
				ann::autoencode::train<DataSetType, TestDataSetType>(trainLog.get(), autoencoder, trainDataset, testDataset, config.epochs, config.learingRate);
			}
			catch(const std::invalid_argument& err)
			{
				Log(Log::ERROR)<<"The models given dont have a compatable output-input size: "<<err.what();
			}
			catch(const ann::ScriptNet::load_errror& err)
			{
				Log(Log::ERROR)<<"Could not load TorchScript model: "<<err.what();
			}
			break;
		}
		case MODE_AUTO:
		{
			Log(Log::INFO)<<"Training autoencoder network with script net loaded from "<<config.scriptPath;
			std::shared_ptr<ann::Net> encoder(new ann::SimpleNet(trainDataset->inputSize(), config.latentSize, 4, 3, false));
			std::shared_ptr<ann::Net> decoder(new ann::SimpleNet(config.latentSize, trainDataset->inputSize(), 4, 3, false));
			std::shared_ptr<ann::AutoEncoder> autoencoder(new ann::AutoEncoder(encoder, decoder));
			ann::autoencode::train<DataSetType, TestDataSetType>(trainLog.get(), autoencoder, trainDataset, testDataset, config.epochs, config.learingRate);
			break;
		}
		case MODE_GAN:
		{
			Log(Log::ERROR)<<"Currently not implemented use earlier version";
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

	if((config.datasetMode != DATASET_DIR_REGRESSION &&
		config.datasetMode != DATASET_TAR_REGRESSION) &&
		(config.mode == MODE_REGRESSION || config.mode == MODE_REGRESSION_SCRIPT))
	{
		Log(Log::ERROR)<<"The regression mode can only be trained with a regression dataset";
		return false;
	}

	if((config.mode == MODE_ANN_SCRIPT || config.mode == MODE_REGRESSION_SCRIPT || config.mode == MODE_AUTO_SCRIPT) && config.scriptPath.empty())
	{
		Log(Log::ERROR)<<"To train a TorchScript a path to a TorchScript must be supplied via -n";
		return false;
	}

	if((config.mode != MODE_ANN_SCRIPT && config.mode != MODE_REGRESSION_SCRIPT && config.mode != MODE_AUTO_SCRIPT) && !config.scriptPath.empty())
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
		case DATASET_DIR:
			return train<EisDirDataset>(config);
		case DATASET_TAR:
			return train<EisTarDataset>(config);
		case DATASET_DIR_REGRESSION:
			return train<RegressionLoaderDir>(config);
		case DATASET_TAR_REGRESSION:
			return train<RegressionLoaderTar>(config);
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
