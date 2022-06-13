#pragma once
#include <string>
#include <fstream>

class LossLog
{
	static constexpr char LOG_DIR[] = "./log";
	std::fstream file;
	size_t iter = 0;

public:
	LossLog(std::string fileName);
	~LossLog();
	bool log(size_t epoch, size_t step, double loss, double acc, size_t total = 0);
};
