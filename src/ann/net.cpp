#include "net.h"
#include "../utils/seqprint.h"
#include <sstream>
#include <fstream>

using namespace ann;

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

	if(node.isMember("outputLabels"))
	{
		Json::Value labels = node["outputLabels"];
		if(!labels.isArray())
		{
			Log(Log::WARN)<<"Could not get output labels for network";
		}
		else
		{
			outputLabels.reserve(labels.size());
			for(Json::ArrayIndex i = 0; i < labels.size(); ++i)
				outputLabels.push_back(labels[i].asString());
		}
	}
}

void ann::Net::getConfiguration(Json::Value& node)
{
	node["inputSize"] = inputSize;
	node["outputSize"] = outputSize;
	node["softmax"] = softmax;
	node["purpose"] = purpose;

	if(!outputLabels.empty())
	{
		Json::Value labels = node["outputLabels"];
		labels.resize(outputLabels.size());
		for(size_t i = 0; i < outputLabels.size(); ++i)
			labels[static_cast<int>(i)] = outputLabels[i];
	}
}

const std::vector<std::string>& ann::Net::getOutputLabels() const
{
	return outputLabels;
}

void ann::Net::setOutputLabels(const std::vector<std::string>& outputLabelsI)
{
	if(static_cast<int64_t>(outputLabels.size()) != outputSize)
		Log(Log::WARN)<<outputLabels.size()<<" output labels where set for a network with a output size of "<<outputSize;

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

static std::pair<size_t, size_t> layerWidthFunction(size_t layer, size_t leyers, size_t input, size_t output)
{
	size_t layerIn = (input*(leyers-layer)/leyers)+output*(layer)/leyers;
	size_t layerOut = (input*(leyers-layer-1)/leyers)+output*(layer+1)/leyers;
	return std::pair<size_t, size_t>(layerIn, layerOut);
}

ann::SimpleNet::SimpleNet(const Json::Value& node):
Net(node)
{
	downsampleSteps = node["downsampleSteps"].asUInt64();
	extraSteps = node["extraSteps"].asUInt64();
	init();
}

ann::SimpleNet::SimpleNet(int64_t inputSizeI, int64_t outputSizeI, size_t downsampleStepsI, size_t extraStepsI, bool softmaxI):
Net(inputSizeI, outputSizeI, softmaxI), downsampleSteps(downsampleStepsI), extraSteps(extraStepsI)
{
	init();
}

void ann::SimpleNet::init()
{
	register_module("SimpleModel", model);

	std::pair<size_t, size_t> sizes = layerWidthFunction(0, downsampleSteps, inputSize, outputSize);
	torch::nn::Linear linear(sizes.first, sizes.first);
	torch::nn::init::xavier_uniform_(linear->weight);
	model->push_back(linear);
	model->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)));
	for(size_t i = 0; i < downsampleSteps-1; ++i)
	{
		sizes = layerWidthFunction(i, downsampleSteps, inputSize, outputSize);
		torch::nn::Linear linear(sizes.first, sizes.second);
		torch::nn::init::xavier_uniform_(linear->weight);
		model->push_back(linear);
		model->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1).inplace(true)));
		if(i % 2 == 0)
			model->push_back(torch::nn::Dropout(torch::nn::DropoutOptions().p(0.2)));
		torch::nn::BatchNorm1d bn(torch::nn::BatchNorm1dOptions(sizes.second).eps(0.5).momentum(0.1).affine(false));
		model->push_back(bn);
		if(i == 0)
		{
			for(size_t j = 0; j < extraSteps; ++j)
			{
				model->push_back(torch::nn::Linear(sizes.second, sizes.second));
				model->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)));
			}
		}
	}
	sizes = layerWidthFunction(downsampleSteps-1, downsampleSteps, inputSize, outputSize);
	model->push_back(torch::nn::Linear(sizes.first, sizes.second));
	if(softmax)
		model->push_back(torch::nn::LogSoftmax(torch::nn::LogSoftmaxOptions(1)));
}

std::shared_ptr<torch::nn::Module> ann::SimpleNet::operator[](size_t index)
{
	return model->operator[](index);
}

torch::Tensor ann::SimpleNet::forward(torch::Tensor x)
{
	return model->forward(x);
}

void ann::SimpleNet::getConfiguration(Json::Value& node)
{
	node["type"] = typeid(*this).name();
	node["downsampleSteps"] = downsampleSteps;
	node["extraSteps"] = extraSteps;
	ann::Net::getConfiguration(node);
}

ann::ConvNet::ConvNet(const Json::Value& node):
Net(node)
{
	downsampleSteps = node["downsampleSteps"].asUInt64();
	extraSteps = node["extraSteps"].asUInt64();
	init();
}

ann::ConvNet::ConvNet(int64_t inputSizeI, int64_t outputSizeI, size_t downsampleStepsI, size_t extraStepsI, bool softmaxI):
Net(inputSizeI, outputSizeI, softmaxI), downsampleSteps(downsampleStepsI), extraSteps(extraStepsI)
{
	init();
}

void ann::ConvNet::init()
{
	int64_t size = inputSize;
	const int conv1Kernel = inputSize/5;
	const int maxPool1Devisor = 2;
	const int conv2Kernel = inputSize/10;
	const int maxPool2Devisor = 2;

	register_module("ConvModel", model);

	//conv part
	//model->push_back(SeqPrint());
	torch::nn::Conv1d conv1d0(torch::nn::Conv1dOptions(1, 3, conv1Kernel).stride(1).bias(false));
	torch::nn::init::xavier_uniform_(conv1d0->weight);
	model->push_back(conv1d0);
	size = size-conv1Kernel+1;
	//model->push_back(SeqPrint());
	model->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)));
	model->push_back(torch::nn::MaxPool1d(torch::nn::MaxPool1dOptions(maxPool1Devisor)));
	size = size/maxPool1Devisor;
	//model->push_back(SeqPrint());
	model->push_back(torch::nn::Conv1d(torch::nn::Conv1dOptions(3, 6, conv2Kernel).stride(1).bias(false)));
	size = size-conv2Kernel+1;
	//model->push_back(SeqPrint());
	model->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)));
	model->push_back(torch::nn::MaxPool1d(torch::nn::MaxPool1dOptions(maxPool2Devisor)));
	size = size/2;
	//model->push_back(SeqPrint());
	model->push_back(torch::nn::Flatten());
	size = size*6;
	//model->push_back(SeqPrint());

	//fully connected layers
	std::pair<size_t, size_t> sizes;
	for(size_t i = 0; i < downsampleSteps-1; ++i)
	{
		torch::nn::BatchNorm1d bn(torch::nn::BatchNorm1dOptions(sizes.second).eps(0.5).momentum(0.1).affine(false));
		model->push_back(bn);

		sizes = layerWidthFunction(i, downsampleSteps, size, outputSize);
		model->push_back(torch::nn::Linear(sizes.first, sizes.second));
		model->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)));

		if(i == 0)
		{
			for(size_t j = 0; j < extraSteps; ++j)
			{
				model->push_back(torch::nn::Linear(sizes.second, sizes.second));
				model->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)));
			}
		}
	}

	sizes = layerWidthFunction(downsampleSteps-1, downsampleSteps, size, outputSize);
	model->push_back(torch::nn::Linear(sizes.first, sizes.second));
	if(softmax)
		model->push_back(torch::nn::LogSoftmax(torch::nn::LogSoftmaxOptions(1)));
}

torch::Tensor ann::ConvNet::forward(torch::Tensor x)
{
	assert(x.dim() < 3);
	if(x.dim() == 2)
		x = x.reshape({x.size(0), 1, x.size(1)});
	else
		x = x.reshape({x.size(0), 1});

	return model->forward(x);
}

std::shared_ptr<torch::nn::Module> ann::ConvNet::operator[](size_t index)
{
	return model->operator[](index);
}

void ann::ConvNet::getConfiguration(Json::Value& node)
{
	node["type"] = typeid(*this).name();
	node["downsampleSteps"] = downsampleSteps;
	node["extraSteps"] = extraSteps;
	ann::Net::getConfiguration(node);
}
