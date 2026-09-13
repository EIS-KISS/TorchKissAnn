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
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "ann/classification.h"
#include "ann/convnet.h"
#include "ann/regression.h"
#include "ann/scriptnet.h"
#include "ann/simplenet.h"
#include "data/eisdataset.h"
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
			ss<<"script="<<scriptPath.filename().string();
		else
			ss<<"network="<<netName()<<",downsampleSteps="<<downsampleSteps<<",extraSteps="<<extraSteps;
		ss<<",learningRate="<<learningRate<<",weightDecay="<<weightDecay;
		return ss.str();
	}
};

struct TrialResult
{
	TrialConfig config;
	double loss = std::numeric_limits<double>::quiet_NaN();
	bool ok = false;
	std::string error;
};

inline ann::regression::regression_loss_type_t regressionLossFromConfig(const TrainLoss loss)
{
	switch(loss)
	{
		case LOSS_SPECTRA:
			return ann::regression::REG_LOSS_EIS;
		case LOSS_MSE:
		default:
			return ann::regression::REG_LOSS_MSE;
	}
}

inline std::vector<TrialConfig> buildTrials(const Config& config, NetKind netKind)
{
	std::vector<TrialConfig> trials;

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
			for(double learningRate : config.learningRates)
				for(double weightDecay : config.weightDecays)
				{
					TrialConfig trial;
					trial.netKind = NET_KIND_SCRIPT;
					trial.scriptPath = script;
					trial.learningRate = learningRate;
					trial.weightDecay = weightDecay;
					trials.push_back(trial);
				}
		return trials;
	}

	for(size_t downsampleSteps : config.downsampleSteps)
		for(size_t extraSteps : config.extraSteps)
			for(double learningRate : config.learningRates)
				for(double weightDecay : config.weightDecays)
				{
					TrialConfig trial;
					trial.netKind = netKind;
					trial.downsampleSteps = downsampleSteps;
					trial.extraSteps = extraSteps;
					trial.learningRate = learningRate;
					trial.weightDecay = weightDecay;
					trials.push_back(trial);
				}
	return trials;
}

inline std::shared_ptr<ann::Net> makeNet(const TrialConfig& trial, bool softmax, int64_t inputSize, int64_t outputSize)
{
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

inline void reportResults(std::vector<TrialResult>& results, const Config& config)
{
	std::sort(results.begin(), results.end(), [](const TrialResult& a, const TrialResult& b)
	{
		if(a.ok != b.ok)
			return a.ok;
		if(!a.ok)
			return false;
		return a.loss < b.loss;
	});

	Log(Log::INFO)<<"======================= Tuning results =======================";
	size_t rank = 0;
	for(const TrialResult& result : results)
	{
		rank++;
		if(result.ok)
			Log(Log::INFO)<<rank<<": loss="<<result.loss<<"  "<<result.config.describe();
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
	csv<<"rank,loss,network,script,downsampleSteps,extraSteps,learningRate,weightDecay,status,error\n";
	rank = 0;
	for(const TrialResult& result : results)
	{
		rank++;
		csv<<rank<<',';
		if(result.ok)
			csv<<result.loss;
		csv<<','<<result.config.netName()<<','<<result.config.scriptPath.filename().string()
			<<','<<result.config.downsampleSteps<<','<<result.config.extraSteps
			<<','<<result.config.learningRate<<','<<result.config.weightDecay
			<<','<<(result.ok ? "ok" : "failed")<<",\""<<result.error<<"\"\n";
	}
}

template <typename TrainNetFn>
static int runTrials(const Config& config, int64_t inputSize, int64_t outputSize, bool softmax, TrainNetFn&& trainNetFn)
{
	std::vector<TrialConfig> trials = buildTrials(config, config.netType == NET_CONV ? NET_KIND_CONV : NET_KIND_SIMPLE);
	if(trials.empty())
	{
		Log(Log::ERROR)<<"Nothing to tune, "<<(config.scriptDir.empty() ? "no configurations were specified" : "no TorchScript models were found in "+config.scriptDir.string());
		return 1;
	}

	Log(Log::INFO)<<"Running "<<trials.size()<<" trials with "<<config.epochs<<" epochs each";

	std::vector<TrialResult> results;
	std::shared_ptr<ann::Net> bestNet;
	double bestLoss = std::numeric_limits<double>::infinity();

	size_t index = 0;
	for(const TrialConfig& trial : trials)
	{
		index++;
		Log(Log::INFO)<<"Trial "<<index<<'/'<<trials.size()<<": "<<trial.describe();

		TrialResult result;
		result.config = trial;
		try
		{
			std::shared_ptr<ann::Net> net = makeNet(trial, softmax, inputSize, outputSize);
			double testLoss = std::numeric_limits<double>::quiet_NaN();
			int ret = trainNetFn(net, trial, &testLoss);
			if(ret != 0)
				result.error = "training failed";
			else if(!std::isfinite(testLoss))
				result.error = "test loss is not finite, did the training diverge?";
			else
			{
				result.ok = true;
				result.loss = testLoss;
				if(testLoss < bestLoss)
				{
					bestLoss = testLoss;
					bestNet = net;
				}
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

	reportResults(results, config);

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
	return runTrials(config, trainDataset->inputSize(), trainDataset->outputSize(), false,
		[&](std::shared_ptr<ann::Net>& net, const TrialConfig& trial, double* testLoss) -> int
		{
			return ann::regression::train(nullptr, net, trainDataset, testDataset, config.epochs,
				trial.learningRate, regressionLossFromConfig(config.loss), nullptr, trial.weightDecay, testLoss);
		});
}

template <typename DatasetType, typename TestDatasetType = DatasetType>
int searchClassification(const Config& config, EisDataset<DatasetType>* trainDataset, EisDataset<TestDatasetType>* testDataset)
{
	return runTrials(config, trainDataset->inputSize(), trainDataset->outputSize(), true,
		[&](std::shared_ptr<ann::Net>& net, const TrialConfig& trial, double* testLoss) -> int
		{
			ann::classification::train(nullptr, net, trainDataset, testDataset, config.epochs,
				trial.learningRate, config.noWeights, trial.weightDecay, testLoss);
			return 0;
		});
}

}
