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

void testLoader(EisGeneratorDataset* data)
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

int main(int argc, char** argv)
{
	Log::level = Log::DEBUG;

	std::cout<<std::setprecision(3)<<std::fixed<<std::setw(3);

	//printDataset(EisGeneratorDataset::getNeisGeneratedDataset(10));
	testfs("./export", "fail_");

}
