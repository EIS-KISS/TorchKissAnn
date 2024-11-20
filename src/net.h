#pragma once

#include <cstdint>
#include <vector>
#include <json/json.h>
#include <filesystem>

#include "log.h"

namespace ann
{

class Net : public torch::nn::Module
{
protected:
	int64_t inputSize;
	int64_t outputSize;
	std::string inputLabel = "EIS";
	torch::Tensor inputFrequencies;
	std::vector<std::string> outputLabels;
	std::string purpose = "Unkown";
	std::vector<std::pair<std::string, int64_t>> extraInputs;
	torch::Tensor outputScalars;
	torch::Tensor outputBiases;
	bool softmax;

public:
	Net(const Json::Value& node);
	Net(int64_t inputSizeI, int64_t outputSizeI, bool softmaxI = true);

	virtual torch::Tensor forward(torch::Tensor x) = 0;

	virtual int64_t getInputSize() const;
	virtual int64_t getOutputSize() const;
	bool hasSoftmaxOutput() const;
	const std::string& getPurpose() const;
	void setPurpose(const std::string& str);
	const std::vector<std::string>& getOutputLabels() const;
	void setOutputLabels(const std::vector<std::string>& outputLabels);
	void setInputLabel(const std::string& inputLabel);
	void setOutputScalars(const torch::Tensor& tensor);
	void setOutputBiases(const torch::Tensor& tensor);
	void setInputFrequencies(const torch::Tensor& tensor);
	torch::Tensor getOutputScalars();
	torch::Tensor getOutputBiases();
	void setExtraInputs(const std::vector<std::pair<std::string, int64_t>>& in);
	std::vector<std::pair<std::string, int64_t>> getExtraInputs();
	virtual void getConfiguration(Json::Value& node);
	virtual bool saveToCheckpointDir(const std::filesystem::path& path);
	virtual bool loadWeightsFromDir(const std::filesystem::path& path);
	virtual std::shared_ptr<torch::nn::Module> operator[](size_t index) = 0;
	static std::shared_ptr<Net> newNetFromConfiguation(const Json::Value& node);
	static std::shared_ptr<Net> newNetFromCheckpointDir(const std::filesystem::path& path);
	template <typename DatasetType> bool setOutputLabelsFromDataset(DatasetType* dataset);
};

template <typename DatasetType> bool Net::setOutputLabelsFromDataset(DatasetType* dataset)
{
	if(getOutputSize() != static_cast<int64_t>(dataset->outputSize()))
		Log(Log::WARN)<<"This dataset has the wrong number of classes, cant set the output lables from this dataset";
	std::vector<std::string> outputLablesTmp(getOutputSize());
	for(int64_t i = 0; i < getOutputSize(); ++i)
	{
		std::string label = dataset->modelStringForClass(i);
		if(label == "Unkown")
		{
			Log(Log::WARN)<<"This dataset dosent provide output labels";
			return false;
		}
		outputLablesTmp[i] = dataset->modelStringForClass(i);
	}
	setOutputLabels(outputLablesTmp);
	return true;
}

}
