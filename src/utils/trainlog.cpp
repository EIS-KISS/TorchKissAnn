#include "trainlog.h"

#include "log.h"
#include "tensoroperators.h"

#include <ctime>

namespace TrainLog
{

static std::filesystem::path logDir;

std::string getDir()
{
	return logDir;
}

void initTrainLog(const std::filesystem::path& dir)
{
	if(!std::filesystem::is_directory(dir))
		std::filesystem::create_directory(dir);

	size_t i = 0;
	while(std::filesystem::exists(dir/("run" + std::to_string(i))))
		++i;

	logDir = dir/("run" + std::to_string(i));
	std::filesystem::create_directory(logDir);
}

void saveMetadata(const MetaData& meta)
{
	if(logDir.empty())
		throw log_error("Trainlog not initalized");
	std::filesystem::path path(logDir/"metadata.csv");

	std::ofstream file(path, std::ios_base::out);
	if(file.is_open())
	{
		file<<std::scientific;
		file<<"start time\t,model,\tnum classes,\tmulticlass,\tlearing rate,\textra layers,\t"
			<<"noise,\tdataset,\tdataset size,\tdataset file,\ttesting dataset file\n";
		file<<time(nullptr)<<",\t"<<meta.model<<",\t"<<meta.classNumber<<",\t"<<meta.multiClass<<",\t"<<meta.learingRate
			<<",\t"<<meta.extraLayers<<",\t"<<meta.noise<<",\t"<<meta.dataset<<",\t"<<meta.datasetSize
			<<",\t"<<meta.trainingFile<<",\t"<<meta.testingFile<<'\n';
		file.close();
	}
	else
	{
		throw log_error("cant open file at " + static_cast<std::string>(path));
	}
}

Loss::Loss(const std::filesystem::path& fileName)
{
	if(logDir.empty())
		throw log_error("Trainlog not initalized");

	std::filesystem::path path(std::string(logDir)/fileName);
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

void Loss::log(size_t epoch, size_t step, double loss, double acc, size_t total, bool print)
{
	if(print)
	{
		Log(Log::INFO, false)<<"Train epoch "<<epoch<<' '<<step;
		if(total > 0)
			Log(Log::INFO, false)<<'/'<<total;
		Log(Log::INFO)<<"\tLoss: "<<loss<<"\tAcc: "<<acc;
	}

	if(!file.is_open())
		throw log_error("log file not open");

	file<<iter++<<','<<epoch<<','<<step<<','<<loss<<','<<acc<<'\n';
	file.flush();
}

Loss::~Loss()
{
	file.close();
}

Tensor::Tensor(const std::string& nameIn): name(nameIn)
{
	if(logDir.empty())
		throw log_error("Trainlog not initalized");
	std::filesystem::path path = std::string(logDir) + "/" + name;
	if(!std::filesystem::is_directory(path))
		std::filesystem::create_directory(path);
}

Tensor::~Tensor(){}

void Tensor::log(const torch::Tensor& tensor)
{
	std::filesystem::path path = std::string(logDir) + "/" + name + "/" + (std::to_string(step)+".csv");
	bool ret = saveTensorToCsv(path, tensor, name+ ", " + std::to_string(step));
	if(!ret)
		throw log_error("Could not save tensor to " + static_cast<std::string>(path));
	++step;
}

}
