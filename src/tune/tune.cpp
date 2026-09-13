//
// TorchKissAnn - A collection of tools to train various types of Machine learning
// algorithms on various types of EIS data
// Copyright (C) 2025 Carl Klemm <carl@uvos.xyz>
//
// This file is part of TorchKissAnn.
//
// TorchKissAnn is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// TorchKissAnn is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with TorchKissAnn.  If not, see <http://www.gnu.org/licenses/>.
//

#include <eisgenerator/log.h>

#include <filesystem>
#include <iostream>
#include <limits>
#include <typeinfo>

#include "ann/classification.h"
#include "ann/regression.h"
#include "data/loaders/dirloader.h"
#include "data/loaders/regressiondirloader.h"
#include "data/loaders/regressionloader.h"
#include "data/loaders/tarloader.h"
#include "data/regressiondataset.h"
#include "globals.h"
#include "log.h"
#include "options.h"
#include "tuner.h"

template <typename DataSetType>
int tuneDataset(const Config& config)
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
			delete testDataset;
			return 1;
		}
	}

	int ret = 0;
	switch(config.mode)
	{
		case MODE_ANN:
		case MODE_ANN_CONV:
			Log(Log::INFO)<<"Tuning a classification network with "<<(config.mode == MODE_ANN_CONV ? "conv" : "simple")<<" net";
			ret = tune::searchClassification<DataSetType, DataSetType>(config, &dataset, testDataset);
			break;

		case MODE_REGRESSION:
		{
			RegressionDataset<DataSetType>* trainDatasetReg = dynamic_cast<RegressionDataset<DataSetType>*>(&dataset);
			RegressionDataset<DataSetType>* testDatasetReg = dynamic_cast<RegressionDataset<DataSetType>*>(testDataset);
			if(!trainDatasetReg || (testDataset && !testDatasetReg))
			{
				Log(Log::ERROR)<<"can not use a dataset of type "<<typeid(dataset).name()<<" to tune regression as this is not a regression dataset";
				ret = 1;
				break;
			}
			Log(Log::INFO)<<"Tuning a regression network with "<<netTypeToStr(config.netType)<<" net";
			ret = tune::searchRegression<DataSetType, DataSetType>(config, trainDatasetReg, testDatasetReg);
			break;
		}

		default:
			Log(Log::ERROR)<<"Tuneing is not yet supported in the mode "<<trainModeToStr(config.mode);
			ret = 3;
			break;
	}

	delete testDataset;
	return ret;
}

bool check_options(const Config& config)
{
	if(config.mode == MODE_INVALID)
	{
		Log(Log::ERROR)<<"You must specify what to tune: -m " MODE_LIST;
		return false;
	}

	if(config.datasetMode == DATASET_INVALID)
	{
		Log(Log::ERROR)<<"You must specify what dataset to use: -d " DATASET_LIST;
		return false;
	}

	if(config.testFileName.empty())
	{
		Log(Log::ERROR)<<"Tuning requires a test dataset against which the configurations can be compared, specify one with -t";
		return false;
	}

	if(config.datasetMode != DATASET_DIR_REGRESSION &&
		config.datasetMode != DATASET_TAR_REGRESSION &&
		config.mode == MODE_REGRESSION)
	{
		Log(Log::ERROR)<<"The regression mode can only be tuned with a regression dataset";
		return false;
	}

	if(config.loss != LOSS_MSE && config.mode != MODE_REGRESSION)
	{
		Log(Log::ERROR)<<"the "<<lossTypeToStr(config.loss)<<" loss can only be used in regression mode";
		return false;
	}

	if(!config.scriptDir.empty())
	{
		if(!std::filesystem::is_directory(config.scriptDir))
		{
			Log(Log::ERROR)<<config.scriptDir.string()<<" is not a directory";
			return false;
		}
		if(!config.downsampleSteps.empty() || !config.extraSteps.empty())
			Log(Log::WARN)<<"--downsample-steps and --extra-steps are ignored when comparing TorchScript models";
	}

	for(double learningRate : config.learningRates)
		if(!(learningRate > 0))
		{
			Log(Log::ERROR)<<"learning rates must be greater than 0";
			return false;
		}

	for(double weightDecay : config.weightDecays)
		if(!(weightDecay >= 0))
		{
			Log(Log::ERROR)<<"weight decays must not be negative";
			return false;
		}

	for(size_t downsampleSteps : config.downsampleSteps)
		if(downsampleSteps < 1)
		{
			Log(Log::ERROR)<<"downsampleSteps must be at least 1";
			return false;
		}

	return true;
}

static void applyDefaults(Config& config)
{
	if(config.downsampleSteps.empty())
		config.downsampleSteps = {4};
	if(config.extraSteps.empty())
		config.extraSteps = {3};
	if(config.learningRates.empty())
		config.learningRates = {0.005};
	if(config.weightDecays.empty())
		config.weightDecays = {0.001};
}

int main(int argc, char** argv)
{
	std::cout<<std::setprecision(5)<<std::fixed<<std::setw(3);
	Log::level = Log::INFO;
	eis::Log::level = eis::Log::ERROR;

	Config config;
	argp_parse(&argp, argc, argv, 0, 0, &config);

	applyDefaults(config);

	choose_device(config.noGpu);
	batch_size = config.batchSize;

	if(!check_options(config))
		return 3;

	Log(Log::INFO)<<"Tuning "<<trainModeToStr(config.mode)<<" on "<<datasetModeToStr(config.datasetMode)<<" dataset";

	switch(config.datasetMode)
	{
		case DATASET_DIR:
			return tuneDataset<EisDirDataset>(config);
		case DATASET_TAR:
			return tuneDataset<EisTarDataset>(config);
		case DATASET_DIR_REGRESSION:
			return tuneDataset<RegressionLoaderDir>(config);
		case DATASET_TAR_REGRESSION:
			return tuneDataset<RegressionLoaderTar>(config);
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
