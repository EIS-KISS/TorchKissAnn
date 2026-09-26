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

#include <cctype>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <typeinfo>

#include "ann/classification.h"
#include "ann/regression.h"
#include "data/loaders/dirloader.h"
#include "data/loaders/eisspectradataset.h"
#include "data/loaders/regressiondirloader.h"
#include "data/loaders/regressionloader.h"
#include "data/loaders/tarloader.h"
#include "data/regressiondataset.h"
#include "globals.h"
#include "log.h"
#include "options.h"
#include "tuner.h"

// Set the canonical frequency grid geometry shared by every trial. Must run before any call to
// inputSize() so the network is sized for the grid rather than for the raw spectra. The per trial
// augmentations are layered on top of this by tune::applyTrialInputPrep().
template <typename DataSetType>
static void applyCanonicalGrid(const Config& config, EisDataset<DataSetType>* dataset, bool training)
{
	if(dataset == nullptr || !config.canonical)
		return;
	EisSpectraDataset* spectra = dynamic_cast<EisSpectraDataset*>(dataset);
	if(spectra == nullptr)
		return;

	CanonicalGrid grid;
	grid.enabled = true;
	grid.fMin = config.canonicalFMin;
	grid.fMax = config.canonicalFMax;
	grid.points = config.canonicalPoints;
	grid.log = config.canonicalLog;
	spectra->setCanonicalGrid(grid);
	spectra->setCanonicalTraining(training);
}

template <typename DataSetType>
static RegressionDataset<DataSetType>* castToRegressionDataset(DataSetType* dataset)
{
	if constexpr(std::is_base_of_v<RegressionDataset<DataSetType>, DataSetType>)
		return dynamic_cast<RegressionDataset<DataSetType>*>(dataset);
	else
		return nullptr;
}

template <typename TrainType, typename TestType>
int tuneDataset(const Config& config)
{
	if(config.fileName.empty())
	{
		Log(Log::ERROR)<<"A valid file or directoy must be specified in conbination with the "<<datasetModeToStr(config.datasetMode)<<" dataset";
		return 1;
	}

	TrainType dataset(config.fileName);
	TestType* testDataset = nullptr;

	if(dataset.size().value() == 0)
	{
		Log(Log::ERROR)<<"Failed to load dataset from "<<config.fileName;
		return 2;
	}

	if(!config.testFileName.empty())
	{
		testDataset = new TestType(config.testFileName);
		if(testDataset->size().value() == 0)
		{
			Log(Log::ERROR)<<"Failed to load dataset from "<<config.testFileName;
			delete testDataset;
			return 2;
		}

		// Loaders intern classes in the order they happen to encounter them, so a training set read
		// from a tar and a test set read from a directory number the same classes differently, and a
		// test set covering a subset of the classes interns into a prefix unrelated to the original
		// numbering. The network's label space is the training dataset's order, so remap the test
		// targets onto it instead of ranking the trials on a permuted label space. Only single label
		// classification can do this: multiclass and regression targets are vectors, not indices.
		if(!dataset.isMulticlass())
		{
			std::vector<std::string> trainLabels;
			for(size_t i = 0; i < dataset.outputSize(); ++i)
				trainLabels.push_back(dataset.outputName(i));

			if(!testDataset->labelsInSameOrderAs(trainLabels))
			{
				if(testDataset->createLabelMapFromLabels(trainLabels))
					Log(Log::INFO)<<"Test dataset interns its "<<testDataset->outputSize()<<" classes in a different order than the training dataset, remapped its targets onto the training label space";
				else
				{
					for(size_t i = 0; i < testDataset->outputSize(); ++i)
						if(std::find(trainLabels.begin(), trainLabels.end(), testDataset->outputName(i)) == trainLabels.end())
							Log(Log::ERROR)<<"The test dataset has the class "<<testDataset->outputName(i)<<" which the training dataset does not contain";
					Log(Log::ERROR)<<"Training and test dataset do not describe the same set of classes. The training dataset has "
						<<dataset.outputSize()<<" classes, the test dataset has "<<testDataset->outputSize();
					delete testDataset;
					return 1;
				}
			}
		}
		else if(dataset.outputSize() != testDataset->outputSize())
		{
			Log(Log::ERROR)<<"Training and test dataset don't have the same number of classes. "
				<<"The traing dataset has "<<dataset.outputSize()
				<<" and the test dataset has "<<testDataset->outputSize();
			delete testDataset;
			return 1;
		}
	}

	// The grid geometry is fixed for the whole search because it sets the input size; the swept
	// augmentations and the |Z| scaling are reconfigured per trial by the search below.
	applyCanonicalGrid(config, &dataset, true);
	applyCanonicalGrid(config, testDataset, false);
	if(config.canonical)
		Log(Log::INFO)<<"Resampling every spectra onto the canonical grid of "<<config.canonicalPoints
			<<(config.canonicalLog ? " log spaced" : " linearly spaced")<<" points from "
			<<config.canonicalFMin<<" to "<<config.canonicalFMax<<" rad/s";

	int ret = 0;
	switch(config.mode)
	{
		case MODE_ANN:
		case MODE_ANN_CONV:
			Log(Log::INFO)<<"Tuning a classification network with "<<(config.mode == MODE_ANN_CONV ? "conv" : "simple")<<" net";
			ret = tune::searchClassification<TrainType, TestType>(config, &dataset, testDataset);
			break;

		case MODE_REGRESSION:
		{
			RegressionDataset<TrainType>* trainDatasetReg = castToRegressionDataset(&dataset);
			RegressionDataset<TestType>* testDatasetReg = castToRegressionDataset(testDataset);
			if(!trainDatasetReg || (testDataset && !testDatasetReg))
			{
				Log(Log::ERROR)<<"can not use a dataset of type "<<typeid(dataset).name()<<" to tune regression as this is not a regression dataset";
				ret = 1;
				break;
			}
			Log(Log::INFO)<<"Tuning a regression network with "<<netTypeToStr(config.netType)<<" net";
			ret = tune::searchRegression<TrainType, TestType>(config, trainDatasetReg, testDatasetReg);
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

	// Same binding as in the train tool: the classification search ranks by classification loss
	// against class ids, the regression search against parameter vectors. Only these two modes are
	// tunable, so every combination is either regression or classification.
	if(config.mode == MODE_REGRESSION && !datasetModeIsRegression(config.datasetMode))
	{
		Log(Log::ERROR)<<"the regression mode can only be tuned with a regression dataset, "
			<<"use -d "<<datasetModeToStr(DATASET_DIR_REGRESSION)<<" or -d "<<datasetModeToStr(DATASET_TAR_REGRESSION)
			<<", not -d "<<datasetModeToStr(config.datasetMode);
		return false;
	}
	if((config.mode == MODE_ANN || config.mode == MODE_ANN_CONV) && datasetModeIsRegression(config.datasetMode))
	{
		Log(Log::ERROR)<<"the "<<trainModeToStr(config.mode)<<" mode can only be tuned with a classification dataset, "
			<<"use -d "<<datasetModeToStr(DATASET_DIR)<<" or -d "<<datasetModeToStr(DATASET_TAR)
			<<", not -d "<<datasetModeToStr(config.datasetMode)
			<<". To tune on parameter targets use -m regression instead.";
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

	SpectraNorm spectraNormMode = SpectraNorm::None;
	for(const std::string& name : config.spectraNorms)
	{
		std::string lowered = name;
		for(char& c : lowered)
			c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
		if(spectraNormFromString(name) == SpectraNorm::None && !lowered.empty() && lowered != "none")
		{
			Log(Log::ERROR)<<"--spectra-norm must be a comma separated list of none, zmean, zgeom and zmax, not "<<name;
			return false;
		}
		spectraNormMode = spectraNormFromString(name);
	}
	if(spectraNormMode != SpectraNorm::None && config.mode == MODE_REGRESSION)
	{
		Log(Log::ERROR)<<"--spectra-norm can not be tuned in regression mode: dividing every spectra by its own "
			<<"impedance level removes exactly what the parameter targets describe.";
		return false;
	}

	for(double augmentShift : config.canonicalAugmentShifts)
		if(!(augmentShift >= 0))
		{
			Log(Log::ERROR)<<"--canonical-augment-shift values must be >= 0";
			return false;
		}
	for(double augmentWindow : config.canonicalAugmentWindows)
		if(!(augmentWindow > 0 && augmentWindow <= 1))
		{
			Log(Log::ERROR)<<"--canonical-augment-window values must be in (0, 1] (fraction of points to keep)";
			return false;
		}

	if(config.canonicalPoints != 0 && config.canonicalPoints < 2)
	{
		Log(Log::ERROR)<<"--canonical-points must be at least 2";
		return false;
	}

	bool augmentationRequested = false;
	for(double augmentShift : config.canonicalAugmentShifts)
		augmentationRequested |= (augmentShift > 0);
	for(double augmentWindow : config.canonicalAugmentWindows)
		augmentationRequested |= (augmentWindow < 1);
	if(augmentationRequested && !config.canonical)
	{
		Log(Log::ERROR)<<"the frequency augmentations operate on the resampled points, so sweeping "
			<<"--canonical-augment-shift or --canonical-augment-window requires the canonical grid: "
			<<"pass --canonical-fmin, --canonical-fmax and --canonical-points";
		return false;
	}
	if(config.canonical && config.mode == MODE_REGRESSION)
	{
		bool shiftRequested = false;
		for(double augmentShift : config.canonicalAugmentShifts)
			shiftRequested |= (augmentShift > 0);
		if(shiftRequested)
		{
			Log(Log::ERROR)<<"frequency shift augmentation can not be tuned in regression mode: shifting a "
				<<"spectra moves the absolute position of the features and invalidates the parameter labels. "
				<<"Window augmentation (--canonical-augment-window) is safe to use here.";
			return false;
		}
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
	// The swept input preparation defaults to the single identity configuration, so a tune that
	// does not ask for these stays exactly as cheap and as reproducible as it was before. Note
	// that _train defaults the shift augmentation to 1 decade; here 0 means off and is what you
	// get unless you name the values you want swept.
	if(config.canonicalAugmentShifts.empty())
		config.canonicalAugmentShifts = {0};
	if(config.canonicalAugmentWindows.empty())
		config.canonicalAugmentWindows = {1};
	if(config.spectraNorms.empty())
		config.spectraNorms = {"none"};

	config.canonical = (config.canonicalFMin > 0 && config.canonicalFMax > config.canonicalFMin);
}

// The training loader is picked by -d. The test loader follows the shape of what is on disk so a
// simulated tar can be scored against a directory of real measurements, which is the whole point of
// using this tool to measure transfer. Regression-ness still follows -d.
template <typename TrainType>
static int dispatchTestDataset(const Config& config)
{
	if(std::filesystem::is_directory(config.testFileName))
		return tuneDataset<TrainType, EisDirDataset>(config);
	return tuneDataset<TrainType, EisTarDataset>(config);
}

template <typename TrainType>
static int dispatchTestRegressionDataset(const Config& config)
{
	if(std::filesystem::is_directory(config.testFileName))
		return tuneDataset<TrainType, RegressionLoaderDir>(config);
	return tuneDataset<TrainType, RegressionLoaderTar>(config);
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
			return dispatchTestDataset<EisDirDataset>(config);
		case DATASET_TAR:
			return dispatchTestDataset<EisTarDataset>(config);
		case DATASET_DIR_REGRESSION:
			return dispatchTestRegressionDataset<RegressionLoaderDir>(config);
		case DATASET_TAR_REGRESSION:
			return dispatchTestRegressionDataset<RegressionLoaderTar>(config);
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
