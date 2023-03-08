#pragma once
#include <string>
#include <fstream>
#include <exception>
#include <filesystem>

constexpr char LOG_DIR[] = "./log";

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

class LossLog
{
	std::fstream file;
	size_t iter = 0;

public:
	LossLog(const std::filesystem::path& fileName);
	~LossLog();
	void log(size_t epoch, size_t step, double loss, double acc, size_t total = 0);
};

class TensorLog
{
	size_t step = 0;
	std::string name;

public:
	TensorLog(const std::string& name);
	~TensorLog();
	void log(const torch::Tensor& tensor);
};
