#include "net.h"
#include "../utils/seqprint.h"
#include <cstdint>
#include <json/value.h>
#include <sstream>
#include <fstream>

#include "tensoroptions.h"
#include "simplenet.h"
#include "convnet.h"

using namespace ann;

ann::Net::Net(int64_t inputSizeI, int64_t outputSizeI, bool softmaxI):
inputSize(inputSizeI), outputSize(outputSizeI), softmax(softmaxI)
{
	outputScalars = torch::ones({outputSize}, tensorOptCpu<float>(false));
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
	outputScalars = torch::ones({outputSize}, tensorOptCpu<float>(false));

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

	if(node.isMember("outputScalars"))
	{
		Json::Value scalars = node["outputLabels"];
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

	Json::Value scalars(Json::ValueType::arrayValue);
	scalars.resize(outputSize);
	assert(outputSize == outputScalars.size(0));
	for(int64_t i = 0; i < outputSize; ++i)
		scalars[static_cast<int>(i)] = outputScalars[i].item().toFloat();
	node["outputScalars"] = scalars;
}

const std::vector<std::string>& ann::Net::getOutputLabels() const
{
	return outputLabels;
}

void ann::Net::setOutputLabels(const std::vector<std::string>& outputLabelsI)
{
	if(static_cast<int64_t>(outputLabelsI.size()) != outputSize)
		Log(Log::WARN)<<"Attempted to set output labels of size "<<outputLabelsI.size()<<" for a network with a output size of "<<outputSize<<" ignoreing";

	outputLabels = outputLabelsI;
}

std::shared_ptr<Net> ann::Net::newNetFromConfiguation(const Json::Value& node)
{
	std::string type = node["type"].asString();

	if(type == typeid(ann::ConvNet).name())
		return std::shared_ptr<Net>(new ann::ConvNet(node));
	else if(type == typeid(ann::SimpleNet).name())
		return std::shared_ptr<Net>(new ann::SimpleNet(node));
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
	torch::load(net, path/"weights.pt");
	return net;
}

const std::string& ann::Net::getPurpose() const
{
	return purpose;
}

void ann::Net::setPurpose(const std::string& str)
{
	purpose = str;
}
