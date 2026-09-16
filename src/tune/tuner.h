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

//! One point of a trial's learning curve, normalised over the two training loops: quality is
//! the test accuracy for classification and the mean R2 for regression, perOutput is the
//! per class accuracy or the R2 of every regression target.
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

	//! True when the regression loop produced the curve, which reports R2 instead of accuracy.
	bool isRegression() const
	{
		return not regressionHistory.empty();
	}

	std::vector<EpochRow> rows() const
	{
		std::vector<EpochRow> rows;
		if(isRegression())
			for(const ann::regression::EpochStats& epoch : regressionHistory)
				rows.push_back(EpochRow{epoch.epoch, epoch.trainLoss, epoch.testLoss,
					epoch.r2Mean, toValueList(epoch.r2)});
		else
			for(const ann::classification::EpochStats& epoch : history)
				rows.push_back(EpochRow{epoch.epoch, epoch.trainLoss, epoch.testLoss,
					epoch.testAccuracy, toValueList(epoch.classAccuracy)});
		return rows;
	}

	//! Index of the epoch with the lowest test loss, 0 if there is no curve.
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
			Log(Log::INFO)<<rank<<": loss="<<result.loss<<" "<<(result.isRegression() ? "r2" : "acc")
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
		<<"parameters,seconds,metric,network,script,downsampleSteps,extraSteps,learningRate,weightDecay,status,error\n";
	rank = 0;
	for(size_t i = 0; i < results.size(); ++i)
	{
		const TrialResult& result = results[i];
		const bool regression = result.isRegression();
		rank++;
		csv<<rank<<','<<i<<',';
		if(result.ok)
			csv<<result.loss<<','<<result.bestEpoch()<<','<<result.bestLoss()<<','
				<<(regression ? std::numeric_limits<double>::quiet_NaN() : result.bestQuality())<<','
				<<(regression ? std::numeric_limits<double>::quiet_NaN() : result.finalQuality())<<','
				<<(regression ? result.bestQuality() : std::numeric_limits<double>::quiet_NaN())<<','
				<<(regression ? result.finalQuality() : std::numeric_limits<double>::quiet_NaN())<<','
				<<result.parameters<<','<<result.seconds;
		csv<<','<<(regression ? "r2" : "accuracy")
			<<','<<result.config.netName()<<','<<result.config.scriptPath.filename().string()
			<<','<<result.config.downsampleSteps<<','<<result.config.extraSteps
			<<','<<result.config.learningRate<<','<<result.config.weightDecay
			<<','<<(result.ok ? "ok" : "failed")<<",\""<<result.error<<"\"\n";
	}
	csv.close();

	std::ofstream curves(config.outputDir/"curves.csv");
	if(curves.is_open())
	{
		curves<<"trial,epoch,trainLoss,testLoss,testAccuracy,r2Mean,network,script,downsampleSteps,extraSteps,learningRate,weightDecay\n";
		for(size_t i = 0; i < results.size(); ++i)
		{
			const TrialResult& result = results[i];
			const bool regression = result.isRegression();
			for(const EpochRow& epoch : result.rows())
				curves<<i<<','<<epoch.epoch<<','<<epoch.trainLoss<<','<<epoch.testLoss<<','
					<<(regression ? std::numeric_limits<double>::quiet_NaN() : epoch.quality)<<','
					<<(regression ? epoch.quality : std::numeric_limits<double>::quiet_NaN())
					<<','<<result.config.netName()<<','<<result.config.scriptPath.filename().string()
					<<','<<result.config.downsampleSteps<<','<<result.config.extraSteps
					<<','<<result.config.learningRate<<','<<result.config.weightDecay<<'\n';
		}
		curves.close();
	}
	else
		Log(Log::WARN)<<"Could not write "<<(config.outputDir/"curves.csv").string();

	std::ofstream classAcc(config.outputDir/"classaccuracy.csv");
	if(classAcc.is_open())
	{
		classAcc<<"trial,epoch,class,accuracy,network,script,learningRate,weightDecay\n";
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
					<<','<<result.config.learningRate<<','<<result.config.weightDecay<<'\n';
		}
		classAcc.close();
	}
	else
		Log(Log::WARN)<<"Could not write "<<(config.outputDir/"classaccuracy.csv").string();

	std::ofstream targetR2(config.outputDir/"targetr2.csv");
	if(targetR2.is_open())
	{
		targetR2<<"trial,epoch,target,r2,network,script,learningRate,weightDecay\n";
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
					<<','<<result.config.learningRate<<','<<result.config.weightDecay<<'\n';
		}
		targetR2.close();
	}
	else
		Log(Log::WARN)<<"Could not write "<<(config.outputDir/"targetr2.csv").string();
}

template <typename TrainNetFn>
static int runTrials(const Config& config, NetKind defaultNetKind, int64_t inputSize, int64_t outputSize, bool softmax, TrainNetFn&& trainNetFn)
{
	std::vector<TrialConfig> trials = buildTrials(config, defaultNetKind);
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
	return runTrials(config, config.netType == NET_CONV ? NET_KIND_CONV : NET_KIND_SIMPLE,
		trainDataset->inputSize(), trainDataset->outputSize(), false,
		[&](std::shared_ptr<ann::Net>& net, const TrialConfig& trial, double* testLoss,
			TrialResult* result) -> int
		{
			return ann::regression::train(nullptr, net, trainDataset, testDataset, config.epochs,
				trial.learningRate, regressionLossFromConfig(config.loss), nullptr, trial.weightDecay, testLoss,
				&result->regressionHistory);
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
		});
}

}
