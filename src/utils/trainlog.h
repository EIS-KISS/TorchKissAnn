#pragma once
#include <string>
#include <fstream>
#include <exception>
#include <filesystem>

#include "ann/net.h"

class TrainLog
{
public:
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

	static std::filesystem::path runsDir;

	std::filesystem::path logDir;
	size_t lossTrainIter = 0;
	size_t lossTestIter = 0;
	std::ofstream lossFileTrain;
	std::ofstream lossFileTest;

	std::ofstream createLossFile(const std::string& fileName);

	void logLoss(std::ofstream& file, size_t epoch, size_t step, double loss, double acc, size_t total, bool print, size_t iteration);

public:
	TrainLog();
	~TrainLog();
	void saveMetadata(const MetaData& meta);

	void logTrainLoss(size_t epoch, size_t step, double loss, double acc, size_t total = 0, bool print = true);
	void logTestLoss(size_t epoch, size_t step, double loss, double acc, size_t total = 0, bool print = true);
	void logTensor(const std::string& name, const torch::Tensor& tensor);
	std::filesystem::path getDir();
	void saveNetwork(std::shared_ptr<ann::Net> net);

	static void setRunsDir(const std::filesystem::path& dir);
	static std::filesystem::path getRunsDir();
};
