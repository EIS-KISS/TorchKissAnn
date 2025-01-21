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

#include "scriptnet.h"

#include <cstdint>
#include <json/reader.h>
#include <string>
#include <torch/csrc/jit/api/module.h>
#include <torch/nn/functional/activation.h>
#include <vector>
#include <json/json.h>
#include <filesystem>
#include <algorithm>

#include "globals.h"

ann::ScriptNet::ScriptNet(const Json::Value& node, bool noload):
Net(node)
{
	if(!noload)
		loadModule(node["module"].asString());
}

ann::ScriptNet::ScriptNet(const std::filesystem::path& scriptPath, bool softmaxI, int64_t inputSizeI, int64_t outputSize):
Net(inputSizeI, outputSize, softmaxI)
{
	loadModule(scriptPath);
}

void ann::ScriptNet::loadModule(const std::filesystem::path& scriptPath)
{
	torch::jit::ExtraFilesMap files;
	files["meta.json"] = "";
	try
	{
		jitModule = torch::jit::load(scriptPath, *offload_device, files);
	}
	catch(torch::Error& err)
	{
		throw load_errror(err.what());
	}

	bool foundMeta = false;
	for(const std::pair<std::string, std::string> file : files)
	{
		if(file.first == "meta.json" && !file.second.empty())
		{
			foundMeta = true;
			Json::Value json;
			Json::Reader reader;
			bool ret = reader.parse(file.second, json);
			if(!ret)
				throw load_errror(scriptPath.string() + " has an invalid meta.json");
			if(json["inputSize"].type() != Json::ValueType::nullValue)
			{
				int64_t scriptInputSize = json["inputSize"].asInt64();
				if(inputSize > 0 && scriptInputSize != inputSize)
					throw load_errror("This script is for input size " +
						std::to_string(scriptInputSize) + " but " +
						std::to_string(inputSize) + " was requested");
				inputSize = scriptInputSize;
			}
			else if(inputSize < 0)
			{
				throw load_errror("Can not load a script with undetermined input size without specifing a input size");
			}
			int64_t scriptOutputSize = json["outputSize"].asInt64();
			if(scriptOutputSize != outputSize && outputSize > 0)
				throw load_errror("The given torch script has a output size incompatable with the given size, script has " +
					std::to_string(scriptOutputSize) + " but " + std::to_string(outputSize) + " was requested");
		}
	}

	if(!foundMeta)
		throw load_errror(scriptPath.string() + " dose not contain meta.json");

	torch::jit::named_parameter_list list = jitModule.named_parameters();

	for(const auto& item : list)
	{
		std::string name = item.name;
		std::replace(name.begin(), name.end(), '.', '-');
		register_parameter(name, item.value);
	}

	loadPath = scriptPath;
}

torch::Tensor ann::ScriptNet::forward(torch::Tensor x)
{
	torch::Tensor output = jitModule.forward({x}).toTensor();

	if(softmax)
		output = torch::nn::functional::log_softmax(output, torch::nn::functional::LogSoftmaxFuncOptions(1));
	return output;
}

bool ann::ScriptNet::saveToCheckpointDir(const std::filesystem::path &path)
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

	Json::FastWriter fastWriter;
	torch::jit::ExtraFilesMap files;
	files["meta.json"] = fastWriter.write(networkMetadata);

	jitModule.to(torch::Device(torch::DeviceType::CPU, 0));
	jitModule.save(path/"module.pt", files);
	jitModule.to(*offload_device);
	return true;
}

bool ann::ScriptNet::loadWeightsFromDir(const std::filesystem::path& path)
{
	loadModule(path/"module.pt");
	return true;
}

void ann::ScriptNet::getConfiguration(Json::Value& node)
{
	Net::getConfiguration(node);
	node["type"] = typeid(*this).name();
	node["module"] = (loadPath/"module.pt").string();
}

void ann::ScriptNet::eval()
{
	train(true);
}

void ann::ScriptNet::train(bool on)
{
	jitModule.train(on);
}
