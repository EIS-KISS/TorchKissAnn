#pragma once
#include <string>
#include <fstream>
#include <exception>
#include <filesystem>


namespace TrainLog
{

class log_error: public std::exception
{
	std::string whatStr;
public:
	log_error(const std::string& whatIn): whatStr(whatIn)
	{}
	virtual const char* what() const noexcept override
	{
		return whatStr.c_str();
	}
};

void initTrainLog(const std::filesystem::path& dir = "./runs");

struct MetaData
{
	std::string model;
	size_t classNumber;
	bool multiClass;
	double learingRate;
	size_t extraLayers;
	double noise;
	std::string dataset;
	size_t datasetSize;
	std::string trainingFile;
	std::string testingFile;
};

void saveMetadata(const MetaData& meta);

std::filesystem::path getDir();

class Loss
{
	std::fstream file;
	size_t iter = 0;

public:
	Loss(const std::filesystem::path& fileName);
	~Loss();
	void log(size_t epoch, size_t step, double loss, double acc, size_t total = 0, bool print = true);
};

class Tensor
{
	size_t step = 0;
	std::string name;

public:
	Tensor(const std::string& name);
	~Tensor();
	void log(const torch::Tensor& tensor);
};
}
