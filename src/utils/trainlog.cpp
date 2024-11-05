#include "trainlog.h"

#include <ctime>
#include <filesystem>
#include <json/json.h>

#include "log.h"
#include "save.h"
#include "gitrev.h"
#include "ploting.h"

std::filesystem::path TrainLog::runsDir;

std::filesystem::path TrainLog::getDir()
{
	return logDir;
}

std::filesystem::path TrainLog::getRunsDir()
{
	return runsDir;
}

void TrainLog::setRunsDir(const std::filesystem::path& dir)
{
	runsDir = dir;
}

std::ofstream TrainLog::createLossFile(const std::string& fileName)
{
	std::filesystem::path lossPath(logDir/fileName);
	std::ofstream file;
	file.open(lossPath, std::ios_base::out);
	if(file.is_open())
	{
		file<<std::scientific;
		file<<"n,epoch,step,loss,acc\n";
	}
	else
	{
		throw log_error("cant open file at " + static_cast<std::string>(lossPath));
	}
	return file;
}

TrainLog::TrainLog()
{
	if(runsDir.empty())
		throw log_error("Trainlog not initalized");
	if(!std::filesystem::is_directory(runsDir))
		std::filesystem::create_directory(runsDir);

	size_t i = 0;
	while(std::filesystem::exists(runsDir/("run" + std::to_string(i))))
		++i;

	logDir = runsDir/("run" + std::to_string(i));
	std::filesystem::create_directory(logDir);

	lossFileTrain = createLossFile("lossTrain.csv");
	lossFileTest = createLossFile("lossValidate.csv");
}

TrainLog::TrainLog(const std::filesystem::path& path)
{
	if(!std::filesystem::is_directory(path))
		std::filesystem::create_directory(path);
	logDir = path;

	lossFileTrain = createLossFile("lossTrain.csv");
	lossFileTest = createLossFile("lossValidate.csv");
}

void TrainLog::saveMetadata(const MetaData& meta)
{
	std::filesystem::path path(logDir/"metadata.json");

	std::ofstream file(path, std::ios_base::out);
	if(file.is_open())
	{
		Json::Value node;
		node["startTime"] = time(nullptr);
		node["gitRevision"] = std::string(git_sha);
		node["model"] = meta.model;
		node["numClasses"] = meta.classNumber;
		node["multiClass"] = meta.multiClass;
		node["learingRate"] = meta.learingRate;
		node["trainingFile"] = meta.trainingFile;
		node["testingFile"] = meta.testingFile;
		file<<node;
		file.close();
	}
	else
	{
		throw log_error("cant open file at " + static_cast<std::string>(path));
	}
}

void TrainLog::logLoss(std::ofstream& file, size_t epoch, size_t step, double loss, double acc, size_t total, bool print, size_t iteration)
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

	file<<iteration<<','<<epoch<<','<<step<<','<<loss<<','<<acc<<'\n';
	file.flush();
}

void TrainLog::logTrainLoss(size_t epoch, size_t step, double loss, double acc, size_t total, bool print)
{
	logLoss(lossFileTrain, epoch, step, loss, acc, total, print, lossTrainIter++);
	trainLossCurve.push_back({lossTrainIter, loss});

	if(plotThread && plotThread->joinable())
		plotThread->join();

	if(lossTrainIter >= 10 && lossTrainIter % 10 == 0)
		plotThread = new std::thread(TrainLog::plot, logDir, trainLossCurve, false);
}

void TrainLog::logTestLoss(size_t epoch, size_t step, double loss, double acc, size_t total, bool print)
{
	logLoss(lossFileTest, epoch, step, loss, acc, total, print, lossTestIter++);
	testLossCurve.push_back({lossTestIter, loss});

	if(plotThread && plotThread->joinable())
		plotThread->join();

	if(lossTrainIter > 3)
		plotThread = new std::thread(TrainLog::plot, logDir, testLossCurve, true);
}

void TrainLog::logTensor(const std::string& name, const torch::Tensor& tensor)
{
	std::filesystem::path dir = logDir/name;
	if(!std::filesystem::is_directory(dir))
		std::filesystem::create_directory(dir);

	std::filesystem::path path;
	size_t i = 0;
	do
	{
		path = dir/std::filesystem::path(std::to_string(i)+".csv");
		++i;
	} while(std::filesystem::exists(path));

	bool ret = csv::save(path, tensor, name+ ", " + std::to_string(i-1));
	if(!ret)
		throw log_error("Could not save tensor to " + static_cast<std::string>(path));
}

void TrainLog::saveNetwork(std::shared_ptr<ann::Net> net, bool finished)
{
	savedCheckpoint = true;
	std::filesystem::path dir;
	if(!finished)
		dir = logDir/std::filesystem::path("checkpoints")/(std::string("checkpoint_")+std::to_string(lossTrainIter));
	else
		dir = logDir/(std::string("finished_network"));

	net->saveToCheckpointDir(dir);
}

void TrainLog::plot(std::filesystem::path logDir, std::vector<std::pair<size_t, double>> loss, bool test)
{
	std::valarray<double> steps(loss.size());
	std::valarray<double> losses(loss.size());
	for(size_t i = 0; i < loss.size(); ++i)
	{
		steps[i] = loss[i].first;
		losses[i] = loss[i].second;
	}

	save2dPlot(test ? logDir/"valLoss.png" : logDir/"trainLoss.png", "steps", "loss", steps, losses, false, true);
}

TrainLog::~TrainLog()
{
	lossFileTrain.close();
	lossFileTest.close();
	if(!savedCheckpoint)
		std::filesystem::remove_all(logDir);
}
