#include <asm-generic/errno.h>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <eisgenerator/model.h>
#include <libsvm/svm.h>
#include <random>
#include <torch/optim.h>
#include <filesystem>

#include "data/generators/eisgendata.h"
#include "log.h"
#include "gan/simplenet.h"
#include "constants.h"
#include "utils/tokenize.h"
#include "data/dataexport.h"
#include "data/generators/classextractordataset.h"
#include "data/filterdata.h"
#include "data/generators/deduplicator.h"
#include "ann/net.h"
#include "utils/tensoroperators.h"

template<typename Dataset>
void testLoader(torch::data::datasets::Dataset<Dataset, torch::data::Example<torch::Tensor, torch::Tensor>>* data)
{
	torch::data::DataLoaderOptions options;
	options = options.batch_size(100);
	options = options.workers(16);
	options = options.max_jobs(32);

	std::cout<<options.batch_size()<<' '<<options.workers()<<' '<<options.max_jobs().value()<<' '<<data->is_stateful<<'\n';
	auto dataLoader = torch::data::make_data_loader(data->map(torch::data::transforms::Stack<>()), options);
	size_t i = 0;
	for(auto& batch : *dataLoader)
	{
		std::cout<<"batchnum: "<<++i<<" size "<<batch.data.numel()<<std::endl;
	}
}

void printDataset(EisGeneratorDataset* eisData)
{
	std::random_device rd;
	std::default_random_engine rande(rd());
	size_t dataSetSize = eisData->size().value();
	for(size_t i = 0; i < 20; ++i)
	{
		size_t index = rande() % dataSetSize;
		std::cout<<"spectrum "<<index<<":\n"<<eisData->get(index).data<<'\n';
	}
}

bool cmpDouble(double a, double b)
{
	double diff = std::fabs(a-b);
	return diff < 0.0001;
}

std::vector<eis::DataPoint> getControll(size_t i, size_t* size = nullptr)
{
	eis::Model model("t{1,2,3}");
	std::vector<eis::Range> modelRange;
	modelRange.push_back(eis::Range(1e3, 1e4, 40, true, 't'));
	modelRange.push_back(eis::Range(1e-6, 1e-5, 40, true, 't'));
	modelRange.push_back(eis::Range(5, 5, 1, true, 't'));
	eis::Range omega(10, 1e6, 100/2, true);
	std::vector<eis::DataPoint> data = model.executeParamByIndex(modelRange, omega, i);
	filterData(data, 100);
	data = torchToEis(eisToTorch(data));
	if(size)
		*size = eis::Model::getRequiredStepsForSweeps(modelRange);
	return data;
}

void diff()
{
	EisGeneratorDataset* testDataset = EisGeneratorDataset::getNeisGeneratedDataset(DATA_WIDTH, 1e4);
	ClassExtractorDataset<EisGeneratorDataset> extractor(testDataset, 5);

	size_t rangeSteps = extractor.size().value();
	for(size_t i = 0; i < rangeSteps; ++i)
	{
		std::vector<eis::DataPoint> controll = getControll(i);

		eis::saveToDisk(controll, std::string("./controll")+std::string("/")+std::to_string(++i)+".csv");
		eis::saveToDisk(controll, std::string("./controll2")+std::string("/")+std::to_string(++i)+".csv");

		torch::Tensor tmp = extractor.get(i).data;
		std::vector<eis::DataPoint> testSweep = torchToEis(tmp);

		eis::saveToDisk(std::vector<eis::DataPoint>(testSweep), std::string("./testSweep")+std::string("/")+std::to_string(++i)+".csv");

		if(controll.size() != testSweep.size())
		{
			Log(Log::ERROR)<<"Sweeps not the same size!";
			return;
		}

		for(size_t j = 0; j < controll.size(); ++j)
		{
			if(!cmpDouble(controll[j].im.real(), testSweep[j].im.real()) || !cmpDouble(controll[j].im.imag(), testSweep[j].im.imag()))
			{
				if(!cmpDouble(controll[j].im.real(), testSweep[j].im.real()))
					Log(Log::ERROR)<<"no real match! at "<<i<<','<<j<<' '<<controll[j].im.real()<<" != "<<testSweep[j].im.real();
				else
					Log(Log::ERROR)<<"no imag match! at "<<i<<','<<j<<' '<<controll[j].im.imag()<<" != "<<testSweep[j].im.imag();
				//return;
			}
		}
		Log(Log::INFO,false)<<'.';

	}
	std::cout<<std::endl;
}

void saveDataset()
{
	EisGeneratorDataset* testDataset = EisGeneratorDataset::getNeisGeneratedDataset(DATA_WIDTH, 1e4);
	ClassExtractorDataset<EisGeneratorDataset> extractor(testDataset, 5);
	saveDataset<ClassExtractorDataset<EisGeneratorDataset>>("./testdatasetexport", &extractor);
}

void testfs(const std::string& str, const std::string& prefix)
{
	const std::filesystem::path directoryPath{str};
	for(const std::filesystem::directory_entry& dirent : std::filesystem::directory_iterator{directoryPath})
	{
		std::string fileName = dirent.path().filename();
		std::cout<<fileName;
		if(dirent.is_regular_file() && fileName.size() > prefix.size() && fileName.find(prefix) == 0)
		{
			fileName.erase(fileName.begin(), fileName.begin()+prefix.size());
			std::vector<std::string> tokens = tokenize(fileName, '_');
			size_t classNum = std::stoul(tokens[0]);

			std::cout<<" is "<<classNum<<'\n';
		}
		else
		{
			std::cout<<" is nothing\n";
		}
	}
}

void testDeduplicator()
{
	EisGeneratorDataset* testDataset = EisGeneratorDataset::getNeisGeneratedDataset(DATA_WIDTH, 1e7);
	DeduplatorDataset dataset(testDataset, testDataset->modelCount());

	std::cout<<"Data:\n"<<dataset.get(42).data<<"\nTarget:\n"<<dataset.get(42).target;

	delete testDataset;
}

int main(int argc, char** argv)
{
	Log::level = Log::DEBUG;

	//std::cout<<std::setprecision(3)<<std::fixed<<std::setw(3);

	torch::Tensor a = torch::tensor({1, 2, 3, 4}, {torch::kFloat64});
	torch::Tensor b =  torch::tensor({2, 4, 5, 6, 7}, {torch::kFloat64});

	std::cout<<toeplitz(a, b);

	return 0;
	torch::Tensor x = torch::randn({1, 100});
	std::cout<<"begin dim "<<x.dim()<<std::endl;
	ann::ConvNet net(100, 6);
	x = net.forward(x);
	std::cout<<"end with "<<x<<std::endl;

	std::cout<<net<<std::endl;

	ann::SimpleNet snet;
	std::cout<<snet<<std::endl;

	//EisGeneratorDataset* testDataset = EisGeneratorDataset::getNeisGeneratedDataset(DATA_WIDTH, 1e7);
	//testLoader<EisGeneratorDataset>(testDataset);

	//testDeduplicator();
	//saveDataset();
	//diff();
	//printDataset(EisGeneratorDataset::getNeisGeneratedDataset(10));
	//testfs("./export", "fail_");
}
