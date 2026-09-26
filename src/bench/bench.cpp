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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <json/json.h>
#include <torch/torch.h>
#include <torch/cuda.h>
#include <torch/csrc/jit/api/module.h>

#include "ann/convnet.h"
#include "ann/scriptnet.h"
#include "ann/simplenet.h"
#include "globals.h"
#include "log.h"
#include "net.h"
#include "options.h"

static void syncDevice()
{
	if(offload_type == torch::kCUDA)
		torch::cuda::synchronize();
}

static void readScriptSizes(const std::filesystem::path& path, int64_t& inputSize, int64_t& outputSize)
{
	torch::jit::ExtraFilesMap files;
	files["meta.json"] = "";
	try
	{
		torch::jit::load(path, torch::kCPU, files);
	}
	catch(torch::Error& err)
	{
		throw ann::ScriptNet::load_errror(err.what());
	}
	for(const std::pair<const std::string, std::string>& file : files)
	{
		if(file.first != "meta.json" or file.second.empty())
			continue;
		Json::Value json;
		Json::Reader reader;
		if(not reader.parse(file.second, json))
			throw ann::ScriptNet::load_errror(path.string()+" has an invalid meta.json");
		if(json["inputSize"].type() != Json::ValueType::nullValue)
			inputSize = json["inputSize"].asInt64();
		if(json["outputSize"].type() != Json::ValueType::nullValue)
			outputSize = json["outputSize"].asInt64();
	}
	if(inputSize < 0 or outputSize < 0)
		throw ann::ScriptNet::load_errror("Could not determine the input/output size of "+path.string()+", pass -i and -O explicitly");
}

static std::shared_ptr<ann::Net> buildNet(const Config& config)
{
	switch(config.netKind)
	{
		case BENCH_NET_CONV:
			return std::shared_ptr<ann::Net>(new ann::ConvNet(config.inputSize, config.outputSize,
						config.downsampleSteps, config.extraSteps, false));
		case BENCH_NET_SCRIPT:
		{
			// the metadata wins where it is present, the requested sizes cover the gaps (the
			// resnet declares no input size in its metadata)
			int64_t inputSize = config.inputSize;
			int64_t outputSize = config.outputSize;
			readScriptSizes(config.scriptPath, inputSize, outputSize);
			return std::shared_ptr<ann::Net>(new ann::ScriptNet(config.scriptPath, false, inputSize, outputSize));
		}
		case BENCH_NET_SIMPLE:
		default:
			return std::shared_ptr<ann::Net>(new ann::SimpleNet(config.inputSize, config.outputSize,
						config.downsampleSteps, config.extraSteps, false));
	}
}

static void runIteration(ann::Net* net, torch::optim::Optimizer& optimizer,
		torch::nn::MSELoss& lossFn, const torch::Tensor& data, const torch::Tensor& targets)
{
	torch::Tensor prediction = net->forward(data);
	torch::Tensor loss = lossFn(prediction, targets);
	optimizer.zero_grad();
	loss.backward();
	optimizer.step();
}

static void writeCsvRow(const Config& config, const std::string& device, int64_t inputSize,
		int64_t outputSize, uint64_t parameters, double meanMs, double medianMs, double minMs,
		double maxMs, double epochSeconds)
{
	bool fresh = not std::filesystem::exists(config.csvPath) or std::filesystem::file_size(config.csvPath) == 0;
	std::ofstream csv(config.csvPath, std::ios_base::app);
	if(!csv.is_open())
	{
		Log(Log::ERROR)<<"Could not open "<<config.csvPath<<" for writing";
		return;
	}
	if(fresh)
		csv<<"device,network,script,inputSize,outputSize,downsampleSteps,extraSteps,batch,parameters,"
			"warmup,iterations,meanMs,medianMs,minMs,maxMs,itersPerSec,samplesPerSec,epochSeconds\n";
	csv<<device<<','<<benchNetName(config.netKind)<<','<<config.scriptPath.filename().string()<<','
		<<inputSize<<','<<outputSize<<','<<config.downsampleSteps<<','<<config.extraSteps<<','
		<<config.batch<<','<<parameters<<','<<config.warmup<<','<<config.iterations<<','
		<<meanMs<<','<<medianMs<<','<<minMs<<','<<maxMs<<','<<(1000.0/meanMs)<<','
		<<(config.batch*1000.0/meanMs)<<',';
	if(config.datasetSize > 0)
		csv<<epochSeconds;
	csv<<'\n';
}

int main(int argc, char** argv)
{
	std::cout<<std::setprecision(3)<<std::fixed;
	Log::level = Log::INFO;

	Config config;
	argp_parse(&argp, argc, argv, 0, 0, &config);

	choose_device(config.noGpu);
	batch_size = config.batch;

	if(config.netKind == BENCH_NET_SCRIPT and config.scriptPath.empty())
	{
		Log(Log::ERROR)<<"-n script requires -s <path to a TorchScript model>";
		return 1;
	}
	if(config.iterations <= 0)
	{
		Log(Log::ERROR)<<"--iterations must be positive";
		return 1;
	}
	if(config.warmup < 0)
	{
		Log(Log::ERROR)<<"--warmup must not be negative";
		return 1;
	}
	if(config.batch < 2)
	{
		Log(Log::ERROR)<<"--batch-size must be at least 2 (BatchNorm needs more than one sample per batch)";
		return 1;
	}

	std::shared_ptr<ann::Net> net;
	try
	{
		net = buildNet(config);
	}
	catch(const ann::ScriptNet::load_errror& err)
	{
		Log(Log::ERROR)<<"Could not load the TorchScript model: "<<err.what();
		return 1;
	}
	if(!net)
	{
		Log(Log::ERROR)<<"Could not build the network";
		return 1;
	}

	net->to(*offload_device);
	net->train();

	// The script net learns its sizes from its metadata, so read them back after construction
	// rather than trusting the requested ones.
	int64_t inputSize = net->getInputSize();
	int64_t outputSize = net->getOutputSize();

	uint64_t parameters = 0;
	for(const torch::Tensor& param : net->parameters())
		parameters += param.numel();

	// One fixed random batch reused every iteration: the shapes and the amount of work are
	// constant, so the timing reflects the architecture and not the data.
	torch::manual_seed(1234);
	torch::Tensor data = torch::rand({config.batch, inputSize}, torch::TensorOptions().device(*offload_device));
	torch::Tensor targets = torch::rand({config.batch, outputSize}, torch::TensorOptions().device(*offload_device));

	torch::optim::AdamW optimizer(net->parameters(), torch::optim::AdamWOptions(0.001).weight_decay(0.001));
	torch::nn::MSELoss lossFn(torch::nn::MSELossOptions().reduction(torch::kMean));

	// The first iterations pay for kernel compilation, autotuning and allocator growth; they are
	// not representative and are discarded.
	for(int i = 0; i < config.warmup; ++i)
		runIteration(net.get(), optimizer, lossFn, data, targets);
	syncDevice();

	std::vector<double> iterationMs;
	iterationMs.reserve(config.iterations);
	for(int i = 0; i < config.iterations; ++i)
	{
		auto start = std::chrono::steady_clock::now();
		runIteration(net.get(), optimizer, lossFn, data, targets);
		syncDevice();
		iterationMs.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
	}

	double sum = 0;
	double minMs = iterationMs.front();
	double maxMs = iterationMs.front();
	for(double ms : iterationMs)
	{
		sum += ms;
		minMs = std::min(minMs, ms);
		maxMs = std::max(maxMs, ms);
	}
	double meanMs = sum / static_cast<double>(iterationMs.size());
	std::sort(iterationMs.begin(), iterationMs.end());
	double medianMs = iterationMs[iterationMs.size()/2];

	double epochSeconds = 0;
	int64_t iterationsPerEpoch = 0;
	if(config.datasetSize > 0)
	{
		iterationsPerEpoch = (config.datasetSize + config.batch - 1) / config.batch;
		epochSeconds = static_cast<double>(iterationsPerEpoch) * meanMs / 1000.0;
	}

	bool gpu = offload_type == torch::kCUDA;
	std::string device = gpu ? "GPU" : "CPU";

	Log(Log::INFO)<<"Device:       "<<device<<(gpu ? "" : " (forced or no GPU available)");
	Log(Log::INFO)<<"Network:      "<<benchNetName(config.netKind)
		<<(config.netKind == BENCH_NET_SCRIPT ? " "+config.scriptPath.filename().string() : "");
	Log(Log::INFO)<<"Shapes:       batch="<<config.batch<<"  input="<<inputSize<<"  output="<<outputSize;
	Log(Log::INFO)<<"Parameters:   "<<parameters;
	Log(Log::INFO)<<"Iterations:   "<<config.warmup<<" warmup + "<<config.iterations<<" timed (forward + backward + step)";
	Log(Log::INFO)<<"Per iteration:  mean="<<meanMs<<" ms  median="<<medianMs<<" ms  min="<<minMs<<" ms  max="<<maxMs<<" ms";
	Log(Log::INFO)<<"Throughput:     "<<(1000.0/meanMs)<<" it/s  "<<(config.batch*1000.0/meanMs)<<" samples/s";
	if(config.datasetSize > 0)
		Log(Log::INFO)<<"Epoch estimate: "<<epochSeconds<<" s  ("<<iterationsPerEpoch<<" iterations for "
			<<config.datasetSize<<" samples)";

	if(not config.csvPath.empty())
		writeCsvRow(config, device, inputSize, outputSize, parameters, meanMs, medianMs, minMs, maxMs, epochSeconds);

	free_device();
	return 0;
}
