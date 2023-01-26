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
#include "data/loaders/deduploader.h"
#include "drt/drt.h"

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

/*void saveDataset()
{
	EisGeneratorDataset* testDataset = EisGeneratorDataset::getNeisGeneratedDataset(DATA_WIDTH, 1e4);
	ClassExtractorDataset<EisGeneratorDataset> extractor(testDataset, 5);
	saveDataset<ClassExtractorDataset<EisGeneratorDataset>>("./testdatasetexport", &extractor);
}*/

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

void testDrt()
{
	eis::Model model("rc");

	std::vector<eis::DataPoint> inSpectra = model.executeSweep(eis::Range(0.001, 1e6, 6000, true));
	std::vector<fvalue> drt = calcDrt(inSpectra);

	std::cout<<"Drt: \n";
	for(const fvalue value : drt)
		std::cout<<value<<'\n';
}

bool testEisGeneratorDataset()
{
	EisGeneratorDataset* dataset = EisGeneratorDataset::newDefaultDataset(100, 0, false);
	Log(Log::INFO)<<__func__<<" size: "<<dataset->size().value();
	Log(Log::INFO)<<__func__<<" classes: "<<dataset->classesCount();

	delete dataset;
	return true;
}

int main(int argc, char** argv)
{
	Log::level = Log::DEBUG;
	testEisGeneratorDataset();
}
