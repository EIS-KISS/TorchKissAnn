#pragma once

#include <cstdint>
#include <vector>
#include <json/json.h>
#include <filesystem>

namespace ann
{

class Net : public torch::nn::Module
{
protected:
	int64_t inputSize;
	int64_t outputSize;
	std::vector<std::string> outputLabels;

public:
	Net(const Json::Value& node);
	Net(int64_t inputSizeI, int64_t outputSizeI):
	inputSize(inputSizeI), outputSize(outputSizeI)
	{}

	virtual torch::Tensor forward(torch::Tensor x) = 0;

	int64_t getInputSize() const;
	int64_t getOutputSize() const;
	const std::vector<std::string>& getOutputLabels() const;
	void setOutputLabels(const std::vector<std::string>& outputLabels);
	virtual void getConfiguration(Json::Value& node);
	bool saveToCheckpointDir(const std::filesystem::path& path);
	virtual std::shared_ptr<torch::nn::Module> operator[](size_t index) = 0;
	static std::shared_ptr<Net> newNetFromConfiguation(const Json::Value& node);
	static std::shared_ptr<Net> newNetFromCheckpointDir(const std::filesystem::path& path);
};

class SimpleNet : public Net
{
	torch::nn::Sequential model;
	size_t downsampleSteps;
	size_t extraSteps;

	void init();

public:
	SimpleNet(const Json::Value& node);
	SimpleNet(int64_t inputSizeI = 100, int64_t outputSizeI = 6, size_t downsampleSteps = 4, size_t extraSteps = 3);
	virtual torch::Tensor forward(torch::Tensor x) override;
	virtual void getConfiguration(Json::Value& node) override;
	virtual std::shared_ptr<torch::nn::Module> operator[](size_t index);
};

class ConvNet : public Net
{
	torch::nn::Sequential model;
	size_t downsampleSteps;
	size_t extraSteps;

	void init();

public:
	ConvNet(const Json::Value& node);
	ConvNet(int64_t inputSizeI = 100, int64_t outputSizeI = 6, size_t downsampleStepsI = 4, size_t extraStepsI = 3);
	virtual torch::Tensor forward(torch::Tensor x) override;
	virtual void getConfiguration(Json::Value& node);
	virtual std::shared_ptr<torch::nn::Module> operator[](size_t index);
};

}
