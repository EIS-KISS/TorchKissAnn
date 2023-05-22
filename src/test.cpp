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
#include "tensoroperators.h"
#include "utils/tokenize.h"
#include "data/dataexport.h"
#include "ann/net.h"
#include "data/loaders/dirloader.h"
#include "data/generators/parameterregressiondataset.h"
#include "tensoroptions.h"

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

bool testEisGeneratorDataset()
{
	EisGeneratorDataset* dataset = EisGeneratorDataset::newDefaultDataset(100, 1e3, 0, false);
	Log(Log::INFO)<<__func__<<" size: "<<dataset->size().value();
	Log(Log::INFO)<<__func__<<" classes: "<<dataset->classesCount();

	delete dataset;
	return true;
}

bool reexportDirDataset()
{
	EisDirDataset dataset("realDataset/train");
	Log(Log::INFO)<<__func__<<" size: "<<dataset.size().value();
	Log(Log::INFO)<<__func__<<" classes: "<<dataset.classesCount();
	torch::Tensor classCounts = dataset.classCounts();
	for(size_t i = 0; i < dataset.classesCount(); ++i)
		Log(Log::INFO)<<__func__<<" class "<<i<<", "<<classCounts[i].item().to<int64_t>()<<" examples: "<<dataset.modelStringForClass(i);
	bool ret = saveDataset<EisDirDataset>("datasetexport", &dataset);
	if(!ret)
		Log(Log::ERROR)<<__func__<<" could not save dataset";
	return ret;
}

void testModelGrad()
{
	ann::SimpleNet net;
	//Log(Log::DEBUG)<<net[0].weights
}

bool testLinearToComplex()
{
	torch::Tensor data = torch::randn({100}, tensorOptCpu<fvalue>());
	torch::Tensor complex = linearToComplex(data);
	if(complex.numel() != data.numel()/2)
		return false;
	torch::Tensor linear = complexToLinear(complex);
	if(torch::sum(data-linear).item().toDouble() > 0.001)
		return false;
	return true;
}

bool testParaDataset()
{
	ParameterRegressionDataset noDrtDds ("r{100}-r{100}c{1e-6}-r{100}c{1e-4}", 100, 0, false);
	ParameterRegressionDataset drtDds("r{100}-r{100}c{1e-6}-r{100}c{1e-4}", 100, 0, true);

	torch::data::Example<torch::Tensor, torch::Tensor> example = noDrtDds.get(23);
	torch::data::Example<torch::Tensor, torch::Tensor> exampleDrt = drtDds.get(23);

	Log(Log::INFO)<<__func__<<" Parameters: "<<example.target;
	Log(Log::INFO)<<__func__<<" Spectra: "<<linearToComplex(example.target);


	return true;
}

int main(int argc, char** argv)
{
	Log::level = Log::DEBUG;
	choose_device(true);

	//testEisGeneratorDataset();
	//reexportDirDataset();

	testLinearToComplex();
	testParaDataset();

	free_device();
	return 0;
}
