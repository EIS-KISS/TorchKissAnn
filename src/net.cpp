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

#include "net.h"
#include <cstdint>
#include <json/value.h>
#include <sstream>
#include <fstream>
#include <torch/csrc/autograd/generated/variable_factories.h>

#include "ann/scriptnet.h"
#include "tensoroptions.h"
#include "ann/simplenet.h"
#include "ann/convnet.h"
#include "ann/autoencoder.h"

using namespace ann;

ann::Net::Net(int64_t inputSizeI, int64_t outputSizeI, bool softmaxI):
inputSize(inputSizeI), outputSize(outputSizeI), softmax(softmaxI)
{
	outputScalars = torch::ones({outputSize}, tensorOptCpu<float>(false));
	outputBiases = torch::zeros({outputSize}, tensorOptCpu<float>(false));
}

int64_t ann::Net::getInputSize() const
{
	return inputSize;
}

int64_t ann::Net::getOutputSize() const
{
	return outputSize;
}

bool ann::Net::hasSoftmaxOutput() const
{
	return softmax;
}

ann::Net::Net(const Json::Value& node)
{
	inputSize = node["inputSize"].asInt64();
	outputSize = node["outputSize"].asInt64();
	softmax = node.get("softmax", "true").asBool();
	purpose = node.get("purpose", "Unkown").asString();
	inputLabel = node.get("inputLabel", "EIS").asString();

	if(node.isMember("outputLabels"))
	{
		Json::Value labels = node["outputLabels"];
		if(!labels.isArray() || labels.size() != outputSize)
		{
			Log(Log::WARN)<<"Could not get output labels for network";
			if(labels.size() != outputSize)
				Log(Log::WARN)<<"The array has the wrong size at "<<labels.size();
		}
		else
		{
			outputLabels.reserve(labels.size());
			for(Json::ArrayIndex i = 0; i < labels.size(); ++i)
				outputLabels.push_back(labels[i].asString());
		}
	}

	if(node.isMember("extraInputs") && !node.isMember("extraInputLengths"))
	{
		Log(Log::WARN)<<"network as extra inputs but not extra input lengths";
	}
	else if(node.isMember("extraInputs") && node.isMember("extraInputLengths"))
	{
		Json::Value inputs = node["extraInputs"];
		Json::Value lengths = node["extraInputLengths"];
		if(!inputs.isArray() || !lengths.isArray() || lengths.size() != inputs.size())
		{
			Log(Log::WARN)<<"Could not get output extra inputs for network";
		}
		else
		{
			for(Json::ArrayIndex i = 0; i < inputs.size(); ++i)
				extraInputs.push_back({inputs.asString(), lengths[i].asInt64()});
		}
	}

	outputScalars = torch::ones({outputSize}, tensorOptCpu<float>(false));
	if(node.isMember("outputScalars"))
	{
		Json::Value scalars = node["outputScalars"];
		if(!scalars.isArray() || scalars.size() != outputSize)
		{
			Log(Log::WARN)<<"Could not get output scalars for network";
			if(scalars.size() != outputSize)
				Log(Log::WARN)<<"The array has the wrong size at "<<scalars.size();
		}
		else
		{
			for(Json::ArrayIndex i = 0; i < scalars.size(); ++i)
				outputScalars[i] = scalars[i].asFloat();
		}
	}

	outputBiases = torch::zeros({outputSize}, tensorOptCpu<float>(false));
	if(node.isMember("outputBiases"))
	{
		Json::Value biases = node["outputBiases"];
		if(!biases.isArray() || biases.size() != outputSize)
		{
			Log(Log::WARN)<<"Could not get output biases for network";
			if(biases.size() != outputSize)
				Log(Log::WARN)<<"The array has the wrong size at "<<biases.size();
		}
		else
		{
			for(Json::ArrayIndex i = 0; i < biases.size(); ++i)
				outputBiases[i] = biases[i].asFloat();
		}
	}

	if(node.isMember("inputFrequencies"))
	{
		Json::Value freqs = node["inputFrequencies"];
		if(!freqs.isArray() || freqs.size() != inputSize)
		{
			Log(Log::WARN)<<"Could not get input frequencies for network";
			if(freqs.size() != inputSize)
				Log(Log::WARN)<<"The array has the wrong size at "<<freqs.size();
		}
		else
		{
			inputFrequencies = torch::zeros({inputSize}, tensorOptCpu<float>(false));
			for(Json::ArrayIndex i = 0; i < freqs.size(); ++i)
				inputFrequencies[i] = freqs[i].asFloat();
		}
	}
}

void ann::Net::setOutputScalars(const torch::Tensor& tensor)
{
	assert(tensor.size(0) == getOutputSize());
	outputScalars = tensor;
}

torch::Tensor ann::Net::getOutputScalars()
{
	return outputScalars;
}

void ann::Net::setOutputBiases(const torch::Tensor& tensor)
{
	assert(tensor.size(0) == getOutputSize());
	outputBiases = tensor;
}

torch::Tensor ann::Net::getOutputBiases()
{
	return outputBiases;
}

void ann::Net::getConfiguration(Json::Value& node)
{
	node["inputSize"] = inputSize;
	node["outputSize"] = outputSize;
	node["softmax"] = softmax;
	node["purpose"] = purpose;
	node["inputLabel"] = inputLabel;

	if(!outputLabels.empty())
	{
		Json::Value labels(Json::ValueType::arrayValue);
		labels.resize(outputLabels.size());
		for(size_t i = 0; i < outputLabels.size(); ++i)
			labels[static_cast<int>(i)] = outputLabels[i];
		node["outputLabels"] = labels;
	}

	if(!extraInputs.empty())
	{
		Json::Value inputs(Json::ValueType::arrayValue);
		Json::Value lengths(Json::ValueType::arrayValue);
		inputs.resize(extraInputs.size());
		lengths.resize(extraInputs.size());
		for(size_t i = 0; i < inputs.size(); ++i)
		{
			inputs[static_cast<int>(i)] = extraInputs[i].first;
			lengths[static_cast<int>(i)] = extraInputs[i].second;
		}
		node["extraInputs"] = inputs;
		node["extraInputLengths"] = lengths;
	}

	Json::Value scalars(Json::ValueType::arrayValue);
	scalars.resize(outputSize);
	assert(outputSize == outputScalars.size(0));
	for(int64_t i = 0; i < outputSize; ++i)
		scalars[static_cast<int>(i)] = outputScalars[i].item().toFloat();
	node["outputScalars"] = scalars;

	Json::Value biases(Json::ValueType::arrayValue);
	biases.resize(outputSize);
	assert(outputSize == outputScalars.size(0));
	for(int64_t i = 0; i < outputSize; ++i)
		biases[static_cast<int>(i)] = outputBiases[i].item().toFloat();
	node["outputBiases"] = biases;

	if(inputFrequencies.numel() != 0)
	{
		assert(inputFrequencies.size(0) == inputSize);
		assert(inputFrequencies.dim() == 1);
		Json::Value frequencies(Json::ValueType::arrayValue);
		frequencies.resize(inputSize);
		for(int64_t i = 0; i < inputSize; ++i)
			biases[static_cast<int>(i)] = inputFrequencies[i].item().toFloat();
		node["inputFrequencies"] = biases;
	}
}

void ann::Net::setInputFrequencies(const torch::Tensor& tensor)
{
	if(tensor.dim() != 1 || tensor.size(0) != inputSize)
	{
		Log(Log::WARN)<<"Attempted to set input frequencies of size "<< tensor.size(0)<<" for a network with an input size of "<<inputSize<<" ignoreing";
		return;
	}

	inputFrequencies = tensor;
}

const std::vector<std::string>& ann::Net::getOutputLabels() const
{
	return outputLabels;
}

void ann::Net::setOutputLabels(const std::vector<std::string>& outputLabelsI)
{
	if(static_cast<int64_t>(outputLabelsI.size()) != outputSize)
	{
		Log(Log::WARN)<<"Attempted to set output labels of size "<<outputLabelsI.size()<<" for a network with a output size of "<<outputSize<<" ignoreing";
		return;
	}

	outputLabels = outputLabelsI;
}

std::shared_ptr<Net> ann::Net::newNetFromConfiguation(const Json::Value& node)
{
	std::string type = node["type"].asString();

	if(type == typeid(ann::ConvNet).name())
		return std::shared_ptr<Net>(new ann::ConvNet(node));
	else if(type == typeid(ann::SimpleNet).name())
		return std::shared_ptr<Net>(new ann::SimpleNet(node));
	else if(type == typeid(ann::ScriptNet).name())
		return std::shared_ptr<Net>(new ann::ScriptNet(node, true));
	else if(type == typeid(ann::ScriptNet).name())
		return std::shared_ptr<Net>(new ann::AutoEncoder(node));
	return nullptr;
}

bool ann::Net::saveToCheckpointDir(const std::filesystem::path &path)
{
	if(!std::filesystem::is_directory(path))
		std::filesystem::create_directories(path);
	if(!std::filesystem::is_directory(path))
		return false;

	Json::Value networkMetadata;
	getConfiguration(networkMetadata);
	std::ofstream networkMetadataFile;
	networkMetadataFile.open(path/"meta.json", std::ios_base::out);
	if(!networkMetadataFile.is_open())
		return false;
	Json::StreamWriterBuilder builder;
	const std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
	writer->write(networkMetadata, &networkMetadataFile);

	networkMetadataFile.close();
	torch::save(shared_from_this(), path/"weights.pt");
	return true;
}

std::shared_ptr<Net> ann::Net::newNetFromCheckpointDir(const std::filesystem::path& path)
{
	if(!std::filesystem::is_directory(path))
		return nullptr;
	std::ifstream file(path/"meta.json", std::ios_base::in);
	if(!file.is_open())
		return nullptr;

	Json::Value json;
	Json::CharReaderBuilder builder;
	JSONCPP_STRING errs;
	bool ret = parseFromStream(builder, file, &json, &errs);
	if(!ret)
		return nullptr;

	std::shared_ptr<Net> net = newNetFromConfiguation(json);
	if(!net)
		return nullptr;
	net->loadWeightsFromDir(path);
	return net;
}

bool ann::Net::loadWeightsFromDir(const std::filesystem::path& path)
{
	std::shared_ptr<Net> net(this, [](void*){});
	torch::load(net, path/"weights.pt");
	return true;
}

const std::string& ann::Net::getPurpose() const
{
	return purpose;
}

void ann::Net::setPurpose(const std::string& str)
{
	purpose = str;
}

void ann::Net::setInputLabel(const std::string& str)
{
	inputLabel = str;
}

void ann::Net::setExtraInputs(const std::vector<std::pair<std::string, int64_t>>& in)
{
	extraInputs = in;
}

std::vector<std::pair<std::string, int64_t>> ann::Net::getExtraInputs()
{
	return extraInputs;
}
