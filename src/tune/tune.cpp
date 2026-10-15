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
#include <cmath>
#include <filesystem>
#include <format>
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
#include "loss/circuitsimplicity.h"
#include "options.h"
#include "tuner.h"


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
static bool applySimplicityLabels(const Config& config, EisDataset<DataSetType>* dataset)
{
	if(dataset == nullptr || config.loss != LOSS_SIMPLICITY)
		return false;

	std::vector<double> omegas = circuit::uniqueOmegas(dataset->frequencies().value_or(torch::Tensor()));
	if(omegas.empty())
	{
		Log(Log::WARN)<<"Dataset does not report a frequency axis, judging element contributions on a fallback grid from 0.01 to 100000 rad/s";
		omegas.resize(32);
		for(size_t i = 0; i < omegas.size(); ++i)
			omegas[i] = std::pow(10.0, -2.0 + 6.0*static_cast<double>(i)/static_cast<double>(omegas.size()-1));
	}

	dataset->enableSimplicityLabels(omegas, config.simplicityPreference, config.simplicityThreshold);
	return dataset->hasSimplicityLabels();
}

static std::string netKindDescription(const Config& config)
{
	if(modeIsScript(config.mode) || !config.scriptPaths.empty() || !config.scriptDir.empty())
		return "script";
	if(config.mode == MODE_ANN_CONV)
		return "conv";
	return netTypeToStr(config.netType);
}

//! "2,4,8" so the swept latent widths fit on one log line.
static std::string joinSizes(const std::vector<size_t>& sizes)
{
	std::string out;
	for(size_t size : sizes)
	{
		if(!out.empty())
			out += ",";
		out += std::to_string(size);
	}
	return out;
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

	// Validate the augmentation model once before the search spends hours on it: the setter reports an
	// unusable string as false, and every trial would silently train without the augmentation.
	if(!config.augmentationModel.empty())
	{
		EisSpectraDataset* trainSpectra = dynamic_cast<EisSpectraDataset*>(&dataset);
		if(trainSpectra == nullptr || !trainSpectra->setAugmentationModel(config.augmentationModel))
		{
			Log(Log::ERROR)<<"--augmentation-model \""<<config.augmentationModel
				<<"\" can not be used on this dataset, see the message above";
			delete testDataset;
			return 1;
		}
		Log(Log::INFO)<<"Augmentation model \""<<config.augmentationModel
			<<"\" active on the training dataset of every trial, leaving "
			<<std::format("{:.0f}", EisSpectraDataset::AUGMENTATION_CLEAN_FRACTION*100)<<"% of the spectra clean";
	}

	bool simplicityLabels = applySimplicityLabels(config, &dataset);
	applySimplicityLabels(config, testDataset);
	if(config.loss == LOSS_SIMPLICITY && not simplicityLabels)
	{
		Log(Log::ERROR)<<"the simplicity loss needs the generating circuit of every spectra, which "<<config.fileName<<" does not carry";
		delete testDataset;
		return 1;
	}

	int ret = 0;
	switch(config.mode)
	{
		case MODE_ANN:
		case MODE_ANN_CONV:
		case MODE_ANN_SCRIPT:
			Log(Log::INFO)<<"Tuning a classification network with "<<netKindDescription(config)<<" net";
			ret = tune::searchClassification<TrainType, TestType>(config, &dataset, testDataset);
			break;

		case MODE_REGRESSION:
		case MODE_REGRESSION_SCRIPT:
		{
			RegressionDataset<TrainType>* trainDatasetReg = castToRegressionDataset(&dataset);
			RegressionDataset<TestType>* testDatasetReg = castToRegressionDataset(testDataset);
			if(!trainDatasetReg || (testDataset && !testDatasetReg))
			{
				Log(Log::ERROR)<<"can not use a dataset of type "<<typeid(dataset).name()<<" to tune regression as this is not a regression dataset";
				ret = 1;
				break;
			}
			Log(Log::INFO)<<"Tuning a regression network with "<<netKindDescription(config)<<" net";
			ret = tune::searchRegression<TrainType, TestType>(config, trainDatasetReg, testDatasetReg);
			break;
		}

		case MODE_AUTO_SCRIPT:
		case MODE_AUTO:
		case MODE_VAE_SCRIPT:
		case MODE_VAE:
			Log(Log::INFO)<<"Tuning a "<<(modeIsVariational(config.mode) ? "variational autoencoder" : "autoencoder")
				<<" with latent size "<<joinSizes(config.latentSizes)<<" and "
				<<(netKindDescription(config))<<" encoder and decoder";
			ret = tune::searchAutoEncoder<TrainType, TestType>(config, &dataset, testDataset);
			break;

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

	if(modeIsRegression(config.mode) && !datasetModeIsRegression(config.datasetMode))
	{
		Log(Log::ERROR)<<"the "<<trainModeToStr(config.mode)<<" mode can only be tuned with a regression dataset, "
			<<"use -d "<<datasetModeToStr(DATASET_DIR_REGRESSION)<<" or -d "<<datasetModeToStr(DATASET_TAR_REGRESSION)
			<<", not -d "<<datasetModeToStr(config.datasetMode);
		return false;
	}
	if(modeIsClassification(config.mode) && datasetModeIsRegression(config.datasetMode))
	{
		Log(Log::ERROR)<<"the "<<trainModeToStr(config.mode)<<" mode can only be tuned with a classification dataset, "
			<<"use -d "<<datasetModeToStr(DATASET_DIR)<<" or -d "<<datasetModeToStr(DATASET_TAR)
			<<", not -d "<<datasetModeToStr(config.datasetMode)
			<<". To tune on parameter targets use -m regression or -m regression_script instead.";
		return false;
	}

	if(lossIsRegressionOnly(config.loss) && !modeIsRegression(config.mode))
	{
		Log(Log::ERROR)<<"the "<<lossTypeToStr(config.loss)<<" loss can only be used in regression mode";
		return false;
	}

	if(lossIsClassificationOnly(config.loss) && !modeIsClassification(config.mode))
	{
		Log(Log::ERROR)<<"the "<<lossTypeToStr(config.loss)<<" loss scores circuit labels, it can only be tuned in the "<<
			trainModeToStr(MODE_ANN)<<", "<<trainModeToStr(MODE_ANN_CONV)<<" and "<<
			trainModeToStr(MODE_ANN_SCRIPT)<<" modes";
		return false;
	}

	if(config.loss == LOSS_SIMPLICITY && config.simplicityThreshold < 0)
	{
		Log(Log::ERROR)<<"--simplicity-threshold must be >= 0";
		return false;
	}

	if(config.loss == LOSS_SIMPLICITY && !(config.simplicityPreference >= 0))
	{
		Log(Log::ERROR)<<"--simplicity-preference must be >= 0, 0 makes the loss indifferent between the accepted circuits";
		return false;
	}

	const bool scriptSource = !config.scriptPaths.empty() || !config.scriptDir.empty();
	if(modeIsScript(config.mode) && !scriptSource)
	{
		Log(Log::ERROR)<<"the "<<trainModeToStr(config.mode)<<" mode tunes TorchScript models, name them with -n or point --scripts at a directory of them";
		return false;
	}
	if(!config.scriptPaths.empty() && !config.scriptDir.empty())
	{
		Log(Log::ERROR)<<"-n and --scripts both name TorchScript models, pick one of them";
		return false;
	}
	if((config.mode == MODE_AUTO || config.mode == MODE_VAE) && scriptSource)
	{
		Log(Log::ERROR)<<trainModeToStr(config.mode)<<" sweeps the taper of native encoder and decoder networks, to tune a TorchScript pair use -m "
			<<(modeIsVariational(config.mode) ? "vae" : "autoencoder");
		return false;
	}

	if(!config.scriptDir.empty())
	{
		if(!std::filesystem::is_directory(config.scriptDir))
		{
			Log(Log::ERROR)<<config.scriptDir.string()<<" is not a directory";
			return false;
		}
		if(modeIsAutoEncoder(config.mode))
		{
			Log(Log::ERROR)<<"--scripts can not be used in the "<<trainModeToStr(config.mode)
				<<" mode: one file in a directory is one network, an autoencoder needs an encoder and a decoder. "
				<<"Name the pairs with -n encoder.pt,decoder.pt and repeat -n to compare several";
			return false;
		}
		if(config.downsampleStepsRequested || config.extraStepsRequested)
			Log(Log::WARN)<<"--downsample-steps and --extra-steps are ignored when comparing TorchScript models";
	}

	for(const std::string& entry : config.scriptPaths)
	{
		size_t count = 0;
		for(const std::string& token : tokenize(entry, ','))
			if(!stripWhitespace(token).empty())
				count++;
		if(count == 0)
		{
			Log(Log::ERROR)<<"-n was given an empty path";
			return false;
		}
		if(modeIsAutoEncoder(config.mode) && count != 2)
		{
			Log(Log::ERROR)<<"in the autoencoder and vae modes -n must name an encoder and a decoder separated by a comma, got: "<<entry;
			return false;
		}
	}

	for(size_t latentSize : config.latentSizes)
		if(latentSize < 1)
		{
			Log(Log::ERROR)<<"--latent-size and --latent-sizes must be at least 1";
			return false;
		}

	if(modeIsAutoEncoder(config.mode) && config.netType == NET_CONV)
	{
		Log(Log::ERROR)<<"--network conv can not be used in the "<<trainModeToStr(config.mode)
			<<" mode: a conv network shortens the sequence instead of lengthening it and can not act as a decoder. "
			<<"Use -m "<<(modeIsVariational(config.mode) ? "vae" : "autoencoder")<<" with -n encoder.pt,decoder.pt and an "
			<<"upsamplenet as the decoder (buildnet.py -t upsamplenet -i <latent size> -o <spectra size>)";
		return false;
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

	for(double spectralWeight : config.spectralWeights)
		if(!(spectralWeight >= 0))
		{
			Log(Log::ERROR)<<"--spectral-weight and --spectral-weights values must not be negative";
			return false;
		}
	bool spectralRequested = false;
	for(double spectralWeight : config.spectralWeights)
		spectralRequested |= (spectralWeight > 0);
	if(spectralRequested && !modeIsAutoEncoder(config.mode))
	{
		Log(Log::ERROR)<<"--spectral-weight can only be tuned in the autoencoder and vae modes: it is an auxiliary "
			<<"reconstruction loss on the frequency axis, which the classification and regression objectives do not use.";
		return false;
	}

	for(double contractiveWeight : config.contractiveWeights)
		if(!(contractiveWeight >= 0))
		{
			Log(Log::ERROR)<<"--contractive-weight and --contractive-weights values must not be negative";
			return false;
		}
	bool contractiveRequested = false;
	for(double contractiveWeight : config.contractiveWeights)
		contractiveRequested |= (contractiveWeight > 0);
	if(contractiveRequested && !modeIsAutoEncoder(config.mode))
	{
		Log(Log::ERROR)<<"--contractive-weight can only be tuned in the autoencoder and vae modes: it regularises "
			<<"the latent space of a reconstruction objective, which the classification and regression objectives do not have.";
		return false;
	}

	for(size_t downsampleSteps : config.downsampleSteps)
		if(downsampleSteps < 1)
		{
			Log(Log::ERROR)<<"downsampleSteps must be at least 1";
			return false;
		}

	bool spectraNormRequested = false;
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
		if(spectraNormFromString(name) != SpectraNorm::None)
			spectraNormRequested = true;
	}
	if(spectraNormRequested && !modeIsClassification(config.mode))
	{
		Log(Log::ERROR)<<"--spectra-norm can only be tuned in the classification modes: dividing every spectra by its own "
			<<"impedance level removes exactly what the regression targets and the autoencoder reconstruction describe.";
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
	if(config.canonical && modeIsRegression(config.mode))
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
	{
		// torchkissann_train trains autoencoders and VAEs with weight decay switched off, so an
		// untuned weight decay has to be 0 here too or the winner would not describe what _train does.
		config.weightDecays = {(modeIsAutoEncoder(config.mode) && !config.weightDecaysRequested) ? 0.0 : 0.001};
	}
	if(config.canonicalAugmentShifts.empty())
		config.canonicalAugmentShifts = {0};
	if(config.canonicalAugmentWindows.empty())
		config.canonicalAugmentWindows = {1};
	if(config.spectraNorms.empty())
		config.spectraNorms = {"none"};
	if(config.latentSizes.empty())
		config.latentSizes = {config.latentSize};
	if(config.spectralWeights.empty())
		config.spectralWeights = {0};
	if(config.contractiveWeights.empty())
		config.contractiveWeights = {0};

	config.canonical = (config.canonicalFMin > 0 && config.canonicalFMax > config.canonicalFMin);
}

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

	progress_bars = !config.noProgress;

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
