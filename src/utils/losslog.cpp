#include "losslog.h"
#include <filesystem>

#include "log.h"

LossLog::LossLog(std::string fileName)
{
	if(!std::filesystem::is_directory(LOG_DIR))
		std::filesystem::create_directory(LOG_DIR);

	std::filesystem::path path(std::string(LOG_DIR) + '/' + fileName);
	file.open(path, std::ios_base::out);
	if(file.is_open())
	{
		file<<std::scientific;
		file<<"n,epoch,step,loss,acc\n";
	}
	else
	{
		Log(Log::WARN)<<"cant open file at "<<path;
	}
}

bool LossLog::log(size_t epoch, size_t step, double loss, double acc, size_t total)
{
	Log(Log::INFO, false)<<"Train epoch "<<epoch<<' '<<step;
	if(total > 0)
		Log(Log::INFO, false)<<'/'<<total;
	Log(Log::INFO)<<"\tLoss: "<<loss<<"\tAcc: "<<acc;

	if(!file.is_open())
		return false;

	file<<iter++<<','<<epoch<<','<<step<<','<<loss<<','<<acc<<'\n';
	file.flush();

	return true;
}

LossLog::~LossLog()
{
	file.close();
}
