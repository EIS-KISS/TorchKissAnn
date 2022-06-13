#include "losslog.h"
#include <filesystem>

LossLog::LossLog(std::string fileName)
{
	if(!std::filesystem::is_directory(LOG_DIR))
		std::filesystem::create_directory(LOG_DIR);
	file.open(std::string(LOG_DIR) + fileName, std::ios_base::out);
	if(file.is_open())
	{
		file<<std::scientific
		file<<"epoch,step,loss,acc\n";
	}
}

bool LossLog::log(size_t epoch, size_t step, double loss, double acc)
{
	if(!file.is_open())
		return false;
	file<<epoch<<','<<step<<','<<loss<<','<<acc<<'\n';
	return true;
}
