/* * TorchKissAnn - A collection of tools to train various types of Machine learning
 * algorithms on various types of EIS data
 * Copyright (C) 2025 Carl Klemm <carl@uvos.xyz>
 *
 * This file is part of TorchKissAnn.
 *
 * TorchKissAnn is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * TorchKissAnn is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with TorchKissAnn.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ann/autoencode.h"
#include "ann/autoencoder.h"
#include "ann/classification.h"
#include "ann/convnet.h"
#include "ann/regression.h"
#include "ann/scriptnet.h"
#include "ann/simplenet.h"
#include "data/eisdataset.h"
#include "data/loaders/eisspectradataset.h"
#include "data/regressiondataset.h"
#include "log.h"
#include "options.h"

namespace tune
{

enum NetKind
{
	NET_KIND_SIMPLE,
	NET_KIND_CONV,
	NET_KIND_SCRIPT
};

struct TrialConfig
{
	NetKind netKind = NET_KIND_SIMPLE;
	std::filesystem::path scriptPath;
	size_t downsampleSteps = 4;
	size_t extraSteps = 3;
	double learningRate = 0.005;
	double weightDecay = 0.001;
	bool autoEncoder = false;
	bool variational = false;
	std::filesystem::path decoderPath;
	size_t latentSize = 10;
	double spectralWeight = 0;
	double contractiveWeight = 0;
	double canonicalAugmentShift = 0;
	double canonicalAugmentWindow = 1;
	std::string spectraNorm = "none";

	std::string netName() const
	{
		switch(netKind)
		{
			case NET_KIND_SCRIPT:
				return "script";
			case NET_KIND_CONV:
				return "conv";
			case NET_KIND_SIMPLE:
			default:
				return "simple";
		}
	}

	std::string describe() const
	{
		std::stringstream ss;
		if(netKind == NET_KIND_SCRIPT)
			ss<<(autoEncoder ? "encoder=" : "script=")<<scriptPath.filename().string();
		else
			ss<<"network="<<netName()<<",downsampleSteps="<<downsampleSteps<<",extraSteps="<<extraSteps;
		if(autoEncoder)
		{
			ss<<",latentSize="<<latentSize;
			if(spectralWeight > 0)
				ss<<",spectralWeight="<<spectralWeight;
			if(contractiveWeight > 0)
				ss<<",contractiveWeight="<<contractiveWeight;
			if(!decoderPath.empty())
				ss<<",decoder="<<decoderPath.filename().string();
			if(variational)
				ss<<",vae";
		}
		ss<<",learningRate="<<learningRate<<",weightDecay="<<weightDecay;
		ss<<",augmentShift="<<canonicalAugmentShift<<",augmentWindow="<<canonicalAugmentWindow
			<<",spectraNorm="<<spectraNorm;
		return ss.str();
	}
};

struct EpochRow
{
	size_t epoch;
	double trainLoss;
	double testLoss;
	double quality;
	std::vector<double> perOutput;
};

inline std::vector<double> toValueList(const torch::Tensor& tensor)
{
	std::vector<double> values;
	if(!tensor.defined())
		return values;
	torch::Tensor flat = tensor.to(torch::kCPU).flatten();
	values.reserve(flat.numel());
	for(int64_t i = 0; i < flat.numel(); ++i)
		values.push_back(flat[i].item<double>());
	return values;
}

struct TrialResult
{
	TrialConfig config;
	double loss = std::numeric_limits<double>::quiet_NaN();
	bool ok = false;
	std::string error;
	uint64_t parameters = 0;
	double seconds = 0;
	std::vector<ann::classification::EpochStats> history;
	std::vector<ann::regression::EpochStats> regressionHistory;
	std::vector<ann::autoencode::EpochStats> autoEncoderHistory;
	std::shared_ptr<ann::Net> net;

	bool isRegression() const
	{
		return not regressionHistory.empty();
	}

	bool isAutoEncoder() const
	{
		return not autoEncoderHistory.empty();
	}

	std::string taskName() const
	{
		if(isAutoEncoder())
			return config.variational ? "vae" : "autoencoder";
		return isRegression() ? "regression" : "classification";
	}

	std::vector<EpochRow> rows() const
	{
		std::vector<EpochRow> rows;
		if(isAutoEncoder())
		{
			for(const ann::autoencode::EpochStats& epoch : autoEncoderHistory)
			{
				rows.push_back(EpochRow{epoch.epoch, epoch.trainLoss, epoch.testLoss,
					epoch.testLoss, toValueList(epoch.mse)});
			}
		}
		else if(isRegression())
		{
			for(const ann::regression::EpochStats& epoch : regressionHistory)
			{
				rows.push_back(EpochRow{epoch.epoch, epoch.trainLoss, epoch.testLoss,
					epoch.r2Mean, toValueList(epoch.r2)});
			}
		}
		else
		{
			for(const ann::classification::EpochStats& epoch : history)
			{
				rows.push_back(EpochRow{epoch.epoch, epoch.trainLoss, epoch.testLoss,
					epoch.testAccuracy, toValueList(epoch.classAccuracy)});
			}
		}
		return rows;
	}

	size_t bestEpoch() const
	{
		std::vector<EpochRow> all = rows();
		size_t best = 0;
		double bestLoss = std::numeric_limits<double>::infinity();
		for(size_t i = 0; i < all.size(); ++i)
			if(std::isfinite(all[i].testLoss) && all[i].testLoss < bestLoss)
			{
				bestLoss = all[i].testLoss;
				best = i;
			}
		return best;
	}

	double bestLoss() const
	{
		std::vector<EpochRow> all = rows();
		if(all.empty())
			return loss;
		return all[bestEpoch()].testLoss;
	}

	double meanLoss() const
	{
		double sum = 0;
		std::vector<EpochRow> all = rows();
		if(all.empty())
			return loss;
		for(const EpochRow& row : all)
			sum += row.testLoss;
		return sum / all.size();
	}

	double bestQuality() const
	{
		std::vector<EpochRow> all = rows();
		if(all.empty())
			return std::numeric_limits<double>::quiet_NaN();
		return all[bestEpoch()].quality;
	}

	double finalQuality() const
	{
		std::vector<EpochRow> all = rows();
		if(all.empty())
			return std::numeric_limits<double>::quiet_NaN();
		return all.back().quality;
	}
};

inline ann::regression::regression_loss_type_t regressionLossFromConfig(const LossType loss)
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

inline std::vector<std::vector<std::filesystem::path>> collectScriptSets(const Config& config)
{
	const bool pair = modeIsAutoEncoder(config.mode);
	std::vector<std::vector<std::filesystem::path>> sets;

	for(const std::string& entry : config.scriptPaths)
	{
		std::vector<std::filesystem::path> paths;
		for(const std::string& token : tokenize(entry, ','))
		{
			std::string stripped = stripWhitespace(token);
			if(!stripped.empty())
				paths.push_back(stripped);
		}
		if(paths.empty())
		{
			continue;
		}
		if(pair && paths.size() != 2)
		{
			throw std::invalid_argument("in the autoencoder and vae modes -n must name an encoder and a decoder separated by a comma, got: "+entry);
		}
		if(pair)
		{
			sets.push_back(paths);
		}
		else
		{
			for(const std::filesystem::path& path : paths)
				sets.push_back(std::vector<std::filesystem::path>{path});
		}
	}

	if(!config.scriptDir.empty())
	{
		std::vector<std::filesystem::path> scripts;
		for(const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(config.scriptDir))
		{
			if(!entry.is_regular_file() || entry.path().extension() != ".pt")
				continue;
			scripts.push_back(entry.path());
		}
		std::sort(scripts.begin(), scripts.end());
		for(const std::filesystem::path& script : scripts)
			sets.push_back(std::vector<std::filesystem::path>{script});
	}

	return sets;
}

inline std::vector<TrialConfig> buildTrials(const Config& config, NetKind netKind)
{
	std::vector<TrialConfig> trials;

	const bool autoMode = modeIsAutoEncoder(config.mode);
	const bool scriptSource = !config.scriptPaths.empty() || !config.scriptDir.empty();

	std::vector<TrialConfig> networks;
	if(scriptSource)
	{
		for(const std::vector<std::filesystem::path>& set : collectScriptSets(config))
		{
			for(size_t latentSize : config.latentSizes)
			{
				TrialConfig network;
				network.netKind = NET_KIND_SCRIPT;
				network.scriptPath = set[0];
				if(set.size() > 1)
					network.decoderPath = set[1];
				network.autoEncoder = autoMode;
				network.variational = modeIsVariational(config.mode);
				network.latentSize = latentSize;
				networks.push_back(network);
			}
		}
	}
	else
	{
		for(size_t downsampleSteps : config.downsampleSteps)
		{
			for(size_t extraSteps : config.extraSteps)
			{
				for(size_t latentSize : config.latentSizes)
				{
					TrialConfig network;
					network.netKind = netKind;
					network.downsampleSteps = downsampleSteps;
					network.extraSteps = extraSteps;
					network.autoEncoder = autoMode;
					network.variational = modeIsVariational(config.mode);
					network.latentSize = latentSize;
					networks.push_back(network);
				}
			}
		}
	}

	for(const TrialConfig& network : networks)
	{
		for(double learningRate : config.learningRates)
		{
			for(double weightDecay : config.weightDecays)
			{
				for(double spectralWeight : config.spectralWeights)
				{
					for(double contractiveWeight : config.contractiveWeights)
					{
						for(double augmentShift : config.canonicalAugmentShifts)
						{
							for(double augmentWindow : config.canonicalAugmentWindows)
							{
								for(const std::string& spectraNorm : config.spectraNorms)
								{
									TrialConfig trial = network;
									trial.learningRate = learningRate;
									trial.weightDecay = weightDecay;
									trial.spectralWeight = spectralWeight;
									trial.contractiveWeight = contractiveWeight;
									trial.canonicalAugmentShift = augmentShift;
									trial.canonicalAugmentWindow = augmentWindow;
									trial.spectraNorm = spectraNorm;
									trials.push_back(trial);
								}
							}
						}
					}
				}
			}
		}
	}
	return trials;
}

inline std::shared_ptr<ann::Net> makeNet(const TrialConfig& trial, bool softmax, int64_t inputSize, int64_t outputSize)
{
	if(trial.autoEncoder)
	{
		// The spectra is its own target, so only the latent width is between the two halves. A
		// variational encoder emits mu and logvar packed into one vector, hence twice the width.
		int64_t encoderOutput = trial.variational ? 2 * static_cast<int64_t>(trial.latentSize) : static_cast<int64_t>(trial.latentSize);
		std::shared_ptr<ann::Net> encoder;
		std::shared_ptr<ann::Net> decoder;
		if(trial.netKind == NET_KIND_SCRIPT)
		{
			encoder.reset(new ann::ScriptNet(trial.scriptPath, false, inputSize, encoderOutput));
			decoder.reset(new ann::ScriptNet(trial.decoderPath, false, trial.latentSize, inputSize));
		}
		else if(trial.netKind == NET_KIND_CONV)
		{
			encoder.reset(new ann::ConvNet(inputSize, encoderOutput, trial.downsampleSteps, trial.extraSteps, false));
			decoder.reset(new ann::ConvNet(trial.latentSize, inputSize, trial.downsampleSteps, trial.extraSteps, false));
		}
		else
		{
			encoder.reset(new ann::SimpleNet(inputSize, encoderOutput, trial.downsampleSteps, trial.extraSteps, false));
			decoder.reset(new ann::SimpleNet(trial.latentSize, inputSize, trial.downsampleSteps, trial.extraSteps, false));
		}
		return std::shared_ptr<ann::Net>(new ann::AutoEncoder(encoder, decoder, trial.variational));
	}

	switch(trial.netKind)
	{
		case NET_KIND_SCRIPT:
			return std::shared_ptr<ann::Net>(new ann::ScriptNet(trial.scriptPath, softmax, inputSize, outputSize));
		case NET_KIND_CONV:
			return std::shared_ptr<ann::Net>(new ann::ConvNet(inputSize, outputSize, trial.downsampleSteps, trial.extraSteps, softmax));
		case NET_KIND_SIMPLE:
		default:
			return std::shared_ptr<ann::Net>(new ann::SimpleNet(inputSize, outputSize, trial.downsampleSteps, trial.extraSteps, softmax));
	}
}

inline double onlyTask(bool matches, double value)
{
	return matches ? value : std::numeric_limits<double>::quiet_NaN();
}

inline void reportResults(std::vector<TrialResult>& results, const Config& config)
{
	Log(Log::INFO)<<"======================= Tuning results =======================";
	size_t rank = 0;
	for(const TrialResult& result : results)
	{
		rank++;
		if(result.ok)
			Log(Log::INFO)<<rank<<": loss="<<result.loss<<" "<<(result.isAutoEncoder() ? "mse" : (result.isRegression() ? "r2" : "acc"))
				<<"="<<result.bestQuality()<<" params="<<result.parameters
				<<"  "<<result.config.describe();
		else
			Log(Log::INFO)<<rank<<": FAILED  "<<result.config.describe()<<"  ("<<result.error<<")";
	}
	Log(Log::INFO)<<"===============================================================";

	if(config.outputDir.empty())
		return;

	std::error_code ec;
	std::filesystem::create_directories(config.outputDir, ec);
	std::ofstream csv(config.outputDir/"results.csv");
	if(!csv.is_open())
	{
		Log(Log::WARN)<<"Could not write "<<(config.outputDir/"results.csv").string();
		return;
	}
	csv<<"rank,trial,loss,bestEpoch,bestLoss,bestAccuracy,finalAccuracy,bestR2,finalR2,"
		<<"parameters,seconds,metric,network,script,downsampleSteps,extraSteps,learningRate,weightDecay,"
		<<"augmentShift,augmentWindow,spectraNorm,task,decoder,latentSize,spectralWeight,contractiveWeight,bestReconMse,finalReconMse,"
		<<"status,error\n";
	rank = 0;
	for(size_t i = 0; i < results.size(); ++i)
	{
		const TrialResult& result = results[i];
		const bool regression = result.isRegression();
		const bool autoEncoder = result.isAutoEncoder();
		rank++;
		csv<<rank<<','<<i<<',';
		if(result.ok)
			csv<<result.loss<<','<<result.bestEpoch()<<','<<result.bestLoss()<<','
				<<onlyTask(not regression && not autoEncoder, result.bestQuality())<<','
				<<onlyTask(not regression && not autoEncoder, result.finalQuality())<<','
				<<onlyTask(regression, result.bestQuality())<<','
				<<onlyTask(regression, result.finalQuality())<<','
				<<result.parameters<<','<<result.seconds;
		csv<<','<<(autoEncoder ? "reconMse" : (regression ? "r2" : "accuracy"))
			<<','<<result.config.netName()<<','<<result.config.scriptPath.filename().string()
			<<','<<result.config.downsampleSteps<<','<<result.config.extraSteps
			<<','<<result.config.learningRate<<','<<result.config.weightDecay
			<<','<<result.config.canonicalAugmentShift<<','<<result.config.canonicalAugmentWindow
			<<','<<result.config.spectraNorm
			<<','<<result.taskName()<<','<<result.config.decoderPath.filename().string()
			<<','<<result.config.latentSize
			<<','<<result.config.spectralWeight
			<<','<<result.config.contractiveWeight
			<<','<<onlyTask(autoEncoder, result.bestQuality())
			<<','<<onlyTask(autoEncoder, result.finalQuality())
			<<','<<(result.ok ? "ok" : "failed")<<",\""<<result.error<<"\"\n";
	}
	csv.close();

	std::ofstream curves(config.outputDir/"curves.csv");
	if(curves.is_open())
	{
		curves<<"trial,epoch,trainLoss,testLoss,testAccuracy,r2Mean,reconMse,network,script,decoder,downsampleSteps,extraSteps,learningRate,weightDecay,augmentShift,augmentWindow,spectraNorm,latentSize,spectralWeight,contractiveWeight\n";
		for(size_t i = 0; i < results.size(); ++i)
		{
			const TrialResult& result = results[i];
			const bool regression = result.isRegression();
			const bool autoEncoder = result.isAutoEncoder();
			for(const EpochRow& epoch : result.rows())
				curves<<i<<','<<epoch.epoch<<','<<epoch.trainLoss<<','<<epoch.testLoss<<','
					<<onlyTask(not regression && not autoEncoder, epoch.quality)<<','
					<<onlyTask(regression, epoch.quality)<<','
					<<onlyTask(autoEncoder, epoch.quality)
					<<','<<result.config.netName()<<','<<result.config.scriptPath.filename().string()
					<<','<<result.config.decoderPath.filename().string()
					<<','<<result.config.downsampleSteps<<','<<result.config.extraSteps
					<<','<<result.config.learningRate<<','<<result.config.weightDecay
					<<','<<result.config.canonicalAugmentShift<<','<<result.config.canonicalAugmentWindow
					<<','<<result.config.spectraNorm<<','<<result.config.latentSize<<','<<result.config.spectralWeight<<','<<result.config.contractiveWeight<<'\n';
		}
		curves.close();
	}
	else
		Log(Log::WARN)<<"Could not write "<<(config.outputDir/"curves.csv").string();

	std::ofstream classAcc(config.outputDir/"classaccuracy.csv");
	if(classAcc.is_open())
	{
		classAcc<<"trial,epoch,class,accuracy,network,script,learningRate,weightDecay,augmentShift,augmentWindow,spectraNorm\n";
		for(size_t i = 0; i < results.size(); ++i)
		{
			const TrialResult& result = results[i];
			if(result.isRegression() || result.history.empty())
				continue;
			const ann::classification::EpochStats& epoch = result.history[result.bestEpoch()];
			if(!epoch.classAccuracy.defined())
				continue;
			torch::Tensor acc = epoch.classAccuracy.to(torch::kCPU);
			acc = acc.flatten();
			for(int64_t c = 0; c < acc.numel(); ++c)
				classAcc<<i<<','<<epoch.epoch<<','<<c<<','<<acc[c].item<double>()
					<<','<<result.config.netName()<<','<<result.config.scriptPath.filename().string()
					<<','<<result.config.learningRate<<','<<result.config.weightDecay
					<<','<<result.config.canonicalAugmentShift<<','<<result.config.canonicalAugmentWindow
					<<','<<result.config.spectraNorm<<'\n';
		}
		classAcc.close();
	}
	else
		Log(Log::WARN)<<"Could not write "<<(config.outputDir/"classaccuracy.csv").string();

	std::ofstream targetR2(config.outputDir/"targetr2.csv");
	if(targetR2.is_open())
	{
		targetR2<<"trial,epoch,target,r2,network,script,learningRate,weightDecay,augmentShift,augmentWindow,spectraNorm\n";
		for(size_t i = 0; i < results.size(); ++i)
		{
			const TrialResult& result = results[i];
			if(!result.isRegression())
				continue;
			const std::vector<EpochRow> all = result.rows();
			if(all.empty())
				continue;
			const EpochRow& epoch = all[result.bestEpoch()];
			for(size_t t = 0; t < epoch.perOutput.size(); ++t)
				targetR2<<i<<','<<epoch.epoch<<','<<t<<','<<epoch.perOutput[t]
					<<','<<result.config.netName()<<','<<result.config.scriptPath.filename().string()
					<<','<<result.config.learningRate<<','<<result.config.weightDecay
					<<','<<result.config.canonicalAugmentShift<<','<<result.config.canonicalAugmentWindow
					<<','<<result.config.spectraNorm<<'\n';
		}
		targetR2.close();
	}
	else
		Log(Log::WARN)<<"Could not write "<<(config.outputDir/"targetr2.csv").string();
}

template <typename DatasetType, typename TestDatasetType>
inline void applyTrialInputPrep(const Config& config, const TrialConfig& trial,
	EisDataset<DatasetType>* trainDataset, EisDataset<TestDatasetType>* testDataset)
{
	EisSpectraDataset* train = dynamic_cast<EisSpectraDataset*>(trainDataset);
	if(train == nullptr)
		return;

	bool regression = modeIsRegression(config.mode);

	CanonicalGrid grid;
	if(config.canonical)
	{
		grid.enabled = true;
		grid.fMin = config.canonicalFMin;
		grid.fMax = config.canonicalFMax;
		grid.points = config.canonicalPoints;
		grid.log = config.canonicalLog;
		// Shifting moves every time constant, which preserves the circuit class but invalidates
		// regression parameter labels. Windowing only crops, so it stays valid for both.
		grid.augmentShift = (trial.canonicalAugmentShift > 0) && !regression;
		grid.augmentMaxShift = trial.canonicalAugmentShift;
		grid.augmentWindow = (trial.canonicalAugmentWindow < 1.0);
		grid.augmentWindowMin = trial.canonicalAugmentWindow;
	}

	SpectraNorm mode = spectraNormFromString(trial.spectraNorm);

	train->setCanonicalGrid(grid);
	train->setCanonicalTraining(true);
	train->setSpectraNorm(mode);
	train->setAugmentationModel(config.augmentationModel);

	if(EisSpectraDataset* test = dynamic_cast<EisSpectraDataset*>(testDataset))
	{
		test->setCanonicalGrid(grid);
		test->setCanonicalTraining(false);
		test->setSpectraNorm(mode);
		test->setAugmentationModel("");
	}
}

template <typename TrainNetFn>
static int runTrials(const Config& config, NetKind defaultNetKind, int64_t inputSize, int64_t outputSize, bool softmax, TrainNetFn&& trainNetFn,
	const std::function<void(const TrialConfig&)>& prepare = std::function<void(const TrialConfig&)>())
{
	std::vector<TrialConfig> trials;
	try
	{
		trials = buildTrials(config, defaultNetKind);
	}
	catch(const std::exception& err)
	{
		Log(Log::ERROR)<<err.what();
		return 1;
	}
	if(trials.empty())
	{
		Log(Log::ERROR)<<"Nothing to tune, "<<((config.scriptDir.empty() && config.scriptPaths.empty())
			? "no configurations were specified"
			: "no TorchScript models were found"+(config.scriptDir.empty() ? std::string() : " in "+config.scriptDir.string()));
		return 1;
	}

	Log(Log::INFO)<<"Running "<<trials.size()<<" trials with "<<config.epochs<<" epochs each";

	std::vector<TrialResult> results;

	size_t index = 0;
	for(const TrialConfig& trial : trials)
	{
		index++;
		Log(Log::INFO)<<"Trial "<<index<<'/'<<trials.size()<<": "<<trial.describe();

		if(prepare)
			prepare(trial);

		TrialResult result;
		result.config = trial;
		try
		{
			std::shared_ptr<ann::Net> net = makeNet(trial, softmax, inputSize, outputSize);
			for(const torch::Tensor& param : net->parameters())
				result.parameters += param.numel();
			double testLoss = std::numeric_limits<double>::quiet_NaN();
			auto start = std::chrono::steady_clock::now();
			int ret = trainNetFn(net, trial, &testLoss, &result);
			result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
			if(ret != 0)
				result.error = "training failed";
			else if(!std::isfinite(testLoss))
				result.error = "test loss is not finite, did the training diverge?";
			else
			{
				result.net = net;
				result.net->to(torch::kCPU);
				result.ok = true;
				result.loss = testLoss;
			}
		}
		catch(const ann::ScriptNet::load_errror& err)
		{
			result.error = std::string("could not load TorchScript: ")+err.what();
		}
		catch(const std::exception& err)
		{
			result.error = err.what();
		}

		if(result.ok)
			Log(Log::INFO)<<"Trial "<<index<<'/'<<trials.size()<<" finished with test loss "<<result.loss;
		else
			Log(Log::WARN)<<"Trial "<<index<<'/'<<trials.size()<<" failed: "<<result.error;

		results.push_back(result);
	}

	std::function<bool(const TrialResult& a, const TrialResult& b)> compare;

	switch(config.metric)
	{
		case METRIC_BEST:
			compare = [](const TrialResult& a, const TrialResult& b)
			{
				if(a.ok != b.ok)
					return a.ok;
				if(!a.ok)
					return false;
				return a.bestLoss() < b.bestLoss();
			};
			break;
		case METRIC_MEAN:
			compare = [](const TrialResult& a, const TrialResult& b)
			{
				if(a.ok != b.ok)
					return a.ok;
				if(!a.ok)
					return false;
				return a.meanLoss() < b.meanLoss();
			};
			break;
		case METRIC_LAST:
		default:
			compare = [](const TrialResult& a, const TrialResult& b)
			{
				if(a.ok != b.ok)
					return a.ok;
				if(!a.ok)
					return false;
				return a.loss < b.loss;
			};
			break;
	};

	std::sort(results.begin(), results.end(), compare);

	reportResults(results, config);

	std::shared_ptr<ann::Net> bestNet;

	if(results.begin()->ok)
		bestNet = results.begin()->net;

	if(bestNet)
	{
		if(config.outputDir.empty())
			Log(Log::INFO)<<"Best configuration: "<<bestNet->getPurpose()<<" ("<<results.front().config.describe()<<")";
		else if(bestNet->saveToCheckpointDir(config.outputDir/"best"))
			Log(Log::INFO)<<"Best configuration: "<<results.front().config.describe()
				<<", network saved to "<<(config.outputDir/"best").string();
		else
			Log(Log::WARN)<<"Could not save the best network to "<<(config.outputDir/"best").string();
	}
	else
		Log(Log::ERROR)<<"No configuration trained successfully, can not determine a best one";

	return bestNet ? 0 : 1;
}

template <typename DatasetType, typename TestDatasetType = DatasetType>
int searchRegression(const Config& config, RegressionDataset<DatasetType>* trainDataset, RegressionDataset<TestDatasetType>* testDataset)
{
	return runTrials(config, config.netType == NET_CONV ? NET_KIND_CONV : NET_KIND_SIMPLE,
		trainDataset->inputSize(), trainDataset->outputSize(), false,
		[&](std::shared_ptr<ann::Net>& net, const TrialConfig& trial, double* testLoss,
			TrialResult* result) -> int
		{
			return ann::regression::train(nullptr, net, trainDataset, testDataset, config.epochs,
				trial.learningRate, regressionLossFromConfig(config.loss), nullptr, trial.weightDecay, testLoss,
				&result->regressionHistory);
		},
		[&](const TrialConfig& trial)
		{
			applyTrialInputPrep(config, trial, trainDataset, testDataset);
		});
}

template <typename DatasetType, typename TestDatasetType = DatasetType>
int searchClassification(const Config& config, EisDataset<DatasetType>* trainDataset, EisDataset<TestDatasetType>* testDataset)
{
	return runTrials(config, config.mode == MODE_ANN_CONV ? NET_KIND_CONV : NET_KIND_SIMPLE,
		trainDataset->inputSize(), trainDataset->outputSize(), true,
		[&](std::shared_ptr<ann::Net>& net, const TrialConfig& trial, double* testLoss,
			TrialResult* result) -> int
		{
			ann::classification::train(nullptr, net, trainDataset, testDataset, config.epochs,
				trial.learningRate, config.noWeights, trial.weightDecay, testLoss, &result->history);
			return 0;
		},
		[&](const TrialConfig& trial)
		{
			applyTrialInputPrep(config, trial, trainDataset, testDataset);
		});
}

template <typename DatasetType, typename TestDatasetType = DatasetType>
int searchAutoEncoder(const Config& config, EisDataset<DatasetType>* trainDataset, EisDataset<TestDatasetType>* testDataset)
{
	return runTrials(config, NET_KIND_SIMPLE,
		trainDataset->inputSize(), trainDataset->inputSize(), false,
		[&](std::shared_ptr<ann::Net>& net, const TrialConfig& trial, double* testLoss,
			TrialResult* result) -> int
		{
			std::shared_ptr<ann::AutoEncoder> autoencoder = std::dynamic_pointer_cast<ann::AutoEncoder>(net);
			ann::autoencode::train(nullptr, autoencoder, trainDataset, testDataset, config.epochs,
				trial.learningRate, config.klBeta, trial.weightDecay, testLoss, &result->autoEncoderHistory, trial.spectralWeight, SpectraNorm::None, trial.contractiveWeight);
			return 0;
		},
		[&](const TrialConfig& trial)
		{
			applyTrialInputPrep(config, trial, trainDataset, testDataset);
		});
}

}
