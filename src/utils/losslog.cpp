#include "losslog.h"

#include "log.h"
#include "tensoroperators.h"

LossLog::LossLog(const std::filesystem::path& fileName)
{
	if(!std::filesystem::is_directory(LOG_DIR))
		std::filesystem::create_directory(LOG_DIR);

	std::filesystem::path path(std::string(LOG_DIR)/fileName);
	file.open(path, std::ios_base::out);
	if(file.is_open())
	{
		file<<std::scientific;
		file<<"n,epoch,step,loss,acc\n";
	}
	else
	{
		throw log_error("cant open file at " + static_cast<std::string>(path));
	}
}

void LossLog::log(size_t epoch, size_t step, double loss, double acc, size_t total)
{
	Log(Log::INFO, false)<<"Train epoch "<<epoch<<' '<<step;
	if(total > 0)
		Log(Log::INFO, false)<<'/'<<total;
	Log(Log::INFO)<<"\tLoss: "<<loss<<"\tAcc: "<<acc;

	if(!file.is_open())
		throw log_error("log file not open");

	file<<iter++<<','<<epoch<<','<<step<<','<<loss<<','<<acc<<'\n';
	file.flush();
}

LossLog::~LossLog()
{
	file.close();
}

TensorLog::TensorLog(const std::string& nameIn): name(nameIn)
{
	std::filesystem::path path = std::string(LOG_DIR) + "/" + name;
	if(!std::filesystem::is_directory(path))
		std::filesystem::create_directory(path);
}

TensorLog::~TensorLog(){}

void TensorLog::log(const torch::Tensor& tensor)
{
	std::filesystem::path path = std::string(LOG_DIR) + "/" + name + "/" + (std::to_string(step)+".csv");
	bool ret = saveTensorToCsv(path, tensor, name);
	if(!ret)
		throw log_error("Could not save tensor to " + static_cast<std::string>(path));
	++step;
}
