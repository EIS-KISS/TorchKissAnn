#pragma once
#include <string>
#include <fstream>

class LossLog
{
	static constexpr char LOGDIR[] = "./log"
	std::fstream file;
	size_t iter = 0;

public:
	LossLog(std::string fileName);
	bool log(size_t epoch, size_t step, double loss, double acc);
}
