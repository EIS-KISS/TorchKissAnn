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

#include <cerrno>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <ios>
#include <iostream>
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
#include "gan/gan.h"
#include "data/loaders/regressiondirloader.h"
#include "data/loaders/tarloader.h"
#include "log.h"
#include "data/regressiondataset.h"
#include "data/loaders/regressionloader.h"
#include "data/loaders/dirloader.h"
#include "data/loaders/eisspectradataset.h"
#include "data/canonicalgrid.h"
#include "options.h"
#include "loss/circuitsimplicity.h"
#include "trainlog.h"
#include "tokenize.h"

template <typename DataSetType>
int train(const Config& config);

template <typename DataSetType>
void trainWithSplitDataset(const Config& config, DataSetType* dataset);

template <typename DataSetType, typename TestDataSetType = DataSetType>
void trainSwitch(const Config& config, EisDataset<DataSetType>* dataset, EisDataset<TestDataSetType>* testDataset);

static ann::regression::regression_loss_type_t regressionLossFromConfig(const LossType loss)
{
	switch(loss)
	{
		case LOSS_SPECTRA:
			return ann::regression::REG_LOSS_EIS;
		case LOSS_PERMUTATION:
			return ann::regression::REG_LOSS_PERMUTATION;
		case LOSS_COMPOSITE:
			return ann::regression::REG_LOSS_COMPOSITE;
		case LOSS_MSE:
		default:
			return ann::regression::REG_LOSS_MSE;
	}
}

// Configure canonical grid resampling on a dataset if the mode supports it. The training dataset
// gets shift augmentation (classification only), the test dataset never does. Must run before any
// call to inputSize() so the network is sized for the grid, not the raw spectra.
template <typename DataSetType>
static void applyCanonicalGrid(const Config& config, EisDataset<DataSetType>* dataset, bool training)
{
	if(dataset == nullptr)
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
	// Shifting scales every time constant by the same factor, which preserves the circuit class but
	// invalidates regression parameter labels, so it is restricted to classification training.
	bool regression = (config.mode == MODE_REGRESSION || config.mode == MODE_REGRESSION_SCRIPT);
	grid.augmentShift = (config.canonicalAugmentMaxShift > 0) && training && !regression;
	grid.augmentMaxShift = config.canonicalAugmentMaxShift;
	// Windowing only crops the measurement, features keep their absolute position, so it is safe for
	// regression as well as classification.
	grid.augmentWindow = (config.canonicalAugmentWindowMin < 1.0) && training;
	grid.augmentWindowMin = config.canonicalAugmentWindowMin;
	spectra->setCanonicalGrid(grid);
	spectra->setCanonicalTraining(training);
}

// Configure the per spectra |Z| scaling on a dataset. Unlike the shift augmentation this is
// deterministic, so the training and the test dataset have to be configured identically or the
// network would be measured against a differently scaled input. Must run before any call to
// inputSize() and before the dataset is copied into a data loader.
template <typename DataSetType>
static void applySpectraNorm(const Config& config, EisDataset<DataSetType>* dataset)
{
	if(dataset == nullptr)
		return;
	SpectraNorm mode = spectraNormFromString(config.spectraNorm);
	if(mode == SpectraNorm::None)
		return;
	EisSpectraDataset* spectra = dynamic_cast<EisSpectraDataset*>(dataset);
	if(spectra == nullptr)
		return;
	spectra->setSpectraNorm(mode);
}

// Build the set valued labels of -s simplicity on a dataset and report whether it produced them.
// Must run after the canonical grid is configured: the band over which an element has to stay out of
// the impedance is the band the network is shown, not the one the file happens to carry. Training and
// test dataset are given the same arguments so both score the same kind of label.
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
		if(testDataset->size().value() == 0)
		{
			Log(Log::ERROR)<<"Failed to load dataset from "<<config.testFileName;
			delete testDataset;
			return 2;
		}

		// Loaders intern classes in the order they happen to encounter them, so a training set read
		// from a tar and a test set read from a directory number the same classes differently, and a
		// test set that is a subset of the training set interns into a prefix unrelated to the
		// original numbering. The network's label space is the training dataset's order, so remap the
		// test targets onto it instead of scoring on a permuted label space. Only single label
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

	applySpectraNorm(config, &dataset);
	applySpectraNorm(config, testDataset);
	if(spectraNormFromString(config.spectraNorm) != SpectraNorm::None)
		Log(Log::INFO)<<"Scaling every spectra to a common impedance level (reference: "<<config.spectraNorm
			<<"), the network sees the shape of the Nyquist plot only";

	if(config.canonical)
	{
		applyCanonicalGrid(config, &dataset, true);
		if(testDataset)
		{
			// Reuse the point count the training dataset resolved from its first spectra so train and
			// test are resampled onto the exact same grid even when --canonical-points was not given.
			EisSpectraDataset* trainSpectra = dynamic_cast<EisSpectraDataset*>(&dataset);
			CanonicalGrid resolved;
			if(trainSpectra && trainSpectra->canonicalGridInfo(resolved))
			{
				Config testConfig = config;
				testConfig.canonicalPoints = resolved.points;
				applyCanonicalGrid(testConfig, testDataset, false);
			}
		}
		EisSpectraDataset* trainSpectra = dynamic_cast<EisSpectraDataset*>(&dataset);
		CanonicalGrid resolved;
		if(trainSpectra && trainSpectra->canonicalGridInfo(resolved))
		{
			std::string aug;
			if(resolved.augmentShift)
				aug += "shift +/- " + std::to_string(resolved.augmentMaxShift) + " decades";
			if(resolved.augmentWindow)
				aug += (aug.empty() ? "" : ", ") + ("window keep >= " + std::to_string(resolved.augmentWindowMin*100) + "%");
			if(aug.empty())
				aug = "none";
			Log(Log::INFO)<<"Canonical grid active: "<<resolved.fMin<<" .. "<<resolved.fMax<<" rad/s, "
				<<resolved.points<<" points ("<<(resolved.log ? "log" : "linear")<<"), augmentation: "<<aug;
		}
	}

	bool simplicityLabels = applySimplicityLabels(config, &dataset);
	applySimplicityLabels(config, testDataset);
	if(config.loss == LOSS_SIMPLICITY && not simplicityLabels)
	{
		Log(Log::ERROR)<<"the simplicity loss needs the generating circuit of every spectra, which "<<config.fileName<<" does not carry";
		if(testDataset)
			delete testDataset;
		return 1;
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
		case MODE_VAE_SCRIPT:
		{
			try
			{
				std::vector<std::string> tokens = tokenize(config.scriptPath, ',');
				if(tokens.size() != 2)
				{
					Log(Log::ERROR)<<"For Autoencoders the network path given by -n must be two paths seperated by a: /path/a,/path/b";
					return;
				}
				Log(Log::INFO)<<"Training variational autoencoder network with script net loaded from "<<config.scriptPath;
				std::shared_ptr<ann::Net> encoder(new ann::ScriptNet(tokens[0], false, trainDataset->inputSize(), 2 * config.latentSize));
				std::shared_ptr<ann::Net> decoder(new ann::ScriptNet(tokens[1], false, config.latentSize, trainDataset->inputSize()));
				std::shared_ptr<ann::AutoEncoder> autoencoder(new ann::AutoEncoder(encoder, decoder, true));
				ann::autoencode::train<DataSetType, TestDataSetType>(trainLog.get(), autoencoder, trainDataset, testDataset, config.epochs, config.learingRate, config.klBeta);
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
		case MODE_VAE:
		{
			Log(Log::INFO)<<"Training variational autoencoder network with simple nets";
			std::shared_ptr<ann::Net> encoder(new ann::SimpleNet(trainDataset->inputSize(), 2 * config.latentSize, 4, 3, false));
			std::shared_ptr<ann::Net> decoder(new ann::SimpleNet(config.latentSize, trainDataset->inputSize(), 4, 3, false));
			std::shared_ptr<ann::AutoEncoder> autoencoder(new ann::AutoEncoder(encoder, decoder, true));
			ann::autoencode::train<DataSetType, TestDataSetType>(trainLog.get(), autoencoder, trainDataset, testDataset, config.epochs, config.learingRate, config.klBeta);
			break;
		}
		case MODE_GAN:
		{
			Log(Log::INFO)<<"Training gan generator and discriminator";
			gan::train(trainLog.get(), trainDataset, config.epochs, config.learingRate);
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
			ann::regression::train(trainLog.get(), net, trainDatasetReg, testDatasetReg, config.epochs, config.learingRate,
				regressionLossFromConfig(config.loss));
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
			ann::regression::train(trainLog.get(), net, trainDatasetReg, testDatasetReg, config.epochs, config.learingRate,
				regressionLossFromConfig(config.loss));
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

	// The supervised modes are bound to one kind of label: classification networks are trained on
	// the interned circuit class, regression networks on the parameter vector. Handing either the
	// other one does not fail until well into the first test cycle, so reject it here. The
	// unsupervised modes (autoencoder, vae, gan) only read the spectra itself and take both.
	bool regressionMode = (config.mode == MODE_REGRESSION || config.mode == MODE_REGRESSION_SCRIPT);
	bool classificationMode = (config.mode == MODE_ANN || config.mode == MODE_ANN_CONV || config.mode == MODE_ANN_SCRIPT);
	if(regressionMode && !datasetModeIsRegression(config.datasetMode))
	{
		Log(Log::ERROR)<<"the "<<trainModeToStr(config.mode)<<" mode can only be trained with a regression dataset, "
			<<"use -d "<<datasetModeToStr(DATASET_DIR_REGRESSION)<<" or -d "<<datasetModeToStr(DATASET_TAR_REGRESSION)
			<<", not -d "<<datasetModeToStr(config.datasetMode);
		return false;
	}
	if(classificationMode && datasetModeIsRegression(config.datasetMode))
	{
		Log(Log::ERROR)<<"the "<<trainModeToStr(config.mode)<<" mode can only be trained with a classification dataset, "
			<<"use -d "<<datasetModeToStr(DATASET_DIR)<<" or -d "<<datasetModeToStr(DATASET_TAR)
			<<", not -d "<<datasetModeToStr(config.datasetMode)
			<<". To train on parameter targets use -m "<<trainModeToStr(MODE_REGRESSION)
			<<" or -m "<<trainModeToStr(MODE_REGRESSION_SCRIPT)<<" instead.";
		return false;
	}

	if((config.mode == MODE_ANN_SCRIPT || config.mode == MODE_REGRESSION_SCRIPT || config.mode == MODE_AUTO_SCRIPT || config.mode == MODE_VAE_SCRIPT) && config.scriptPath.empty())
	{
		Log(Log::ERROR)<<"To train a TorchScript a path to a TorchScript must be supplied via -n";
		return false;
	}

	if((config.mode != MODE_ANN_SCRIPT && config.mode != MODE_REGRESSION_SCRIPT && config.mode != MODE_AUTO_SCRIPT && config.mode != MODE_VAE_SCRIPT) && !config.scriptPath.empty())
	{
		Log(Log::ERROR)<<"a path to a TorchScript can only be supplied in "<<trainModeToStr(MODE_ANN_SCRIPT)<<" mode";
		return false;
	}

	if(lossIsRegressionOnly(config.loss) && !modeIsRegression(config.mode))
	{
		Log(Log::ERROR)<<"the "<<lossTypeToStr(config.loss)<<" loss can only be used in regression mode";
		return false;
	}

	if(config.loss == LOSS_SIMPLICITY && !modeIsClassification(config.mode))
	{
		Log(Log::ERROR)<<"the "<<lossTypeToStr(config.loss)<<" loss can only be used in "<<
			trainModeToStr(MODE_ANN)<<", "<<trainModeToStr(MODE_ANN_CONV)<<" and "<<
			trainModeToStr(MODE_ANN_SCRIPT)<<" mode";
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

	bool canonicalRequested = (config.canonicalFMin > 0) || (config.canonicalFMax > 0);
	if(canonicalRequested)
	{
		bool supportedMode = (config.mode == MODE_ANN || config.mode == MODE_ANN_CONV || config.mode == MODE_ANN_SCRIPT ||
			config.mode == MODE_REGRESSION || config.mode == MODE_REGRESSION_SCRIPT);
		if(!supportedMode)
		{
			Log(Log::ERROR)<<"the canonical grid is only supported in ann, conv, script, regression and regression_script modes";
			return false;
		}
		if(!(config.canonicalFMin > 0 && config.canonicalFMax > config.canonicalFMin))
		{
			Log(Log::ERROR)<<"the canonical grid needs both --canonical-fmin and --canonical-fmax with 0 < fmin < fmax";
			return false;
		}
	}
	if(config.canonicalPoints != 0 && config.canonicalPoints < 2)
	{
		Log(Log::ERROR)<<"--canonical-points must be at least 2";
		return false;
	}
	if(config.canonicalAugmentMaxShift < 0)
	{
		Log(Log::ERROR)<<"--canonical-augment-shift must be >= 0";
		return false;
	}
	if(config.canonicalAugmentWindowMin <= 0 || config.canonicalAugmentWindowMin > 1)
	{
		Log(Log::ERROR)<<"--canonical-augment-window must be in (0, 1] (fraction of points to keep)";
		return false;
	}

	SpectraNorm spectraNormMode = spectraNormFromString(config.spectraNorm);
	std::string spectraNormName = config.spectraNorm;
	for(char& c : spectraNormName)
		c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
	if(spectraNormMode == SpectraNorm::None && !spectraNormName.empty() && spectraNormName != "none")
	{
		Log(Log::ERROR)<<"--spectra-norm must be one of none, zmean, zgeom or zmax, not "<<config.spectraNorm;
		return false;
	}
	if(spectraNormMode != SpectraNorm::None && !classificationMode)
	{
		Log(Log::ERROR)<<"--spectra-norm can only be used in the classification modes "<<trainModeToStr(MODE_ANN)<<", "
			<<trainModeToStr(MODE_ANN_CONV)<<" and "<<trainModeToStr(MODE_ANN_SCRIPT)<<": dividing every spectra by its "
			<<"own impedance level removes exactly what the regression targets, the autoencoder reconstruction and the "
			<<"gan discriminator depend on.";
		return false;
	}
	if(canonicalRequested && config.canonicalAugmentShiftRequested && config.canonicalAugmentMaxShift > 0 &&
		(config.mode == MODE_REGRESSION || config.mode == MODE_REGRESSION_SCRIPT))
	{
		Log(Log::WARN)<<"frequency shift augmentation is ignored in regression mode: shifting a spectra "
			<<"changes the absolute position of the features and invalidates the parameter labels. "
			<<"Window augmentation (--canonical-augment-window) is safe to use here.";
	}

	return true;
}

int main(int argc, char** argv)
{
	std::cout<<std::setprecision(5)<<std::fixed<<std::setw(3);
	Log::level = Log::INFO;

	Config config;
	argp_parse(&argp, argc, argv, 0, 0, &config);

	// Canonical mode is on exactly when both grid edges were supplied (validated in check_options).
	config.canonical = (config.canonicalFMin > 0 && config.canonicalFMax > config.canonicalFMin);

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
