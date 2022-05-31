#include <cstddef>
#include <iostream>
#include <libsvm/svm.h>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>

#include "utils/log.h"
#include "options.h"
#include "data/loaders/csv.h"
#include "data/loaders/mpt.h"
#include "data/generators/trashgenerator.h"
#include "data/eistotorch.h"
#include "data/generators/eisgendata.h"
#include "nearest/nearest.h"
#include "utils/log.h"
#include "svm/svm.h"
#include "ann/ann.h"
#include "data/print.h"

#include "constants.h"

bool yesNoPrompt(std::string msg)
{
	while(true)
	{
		std::cout<<msg<<" [y/N]\n";
		std::string resp;
		std::cin>>resp;
		if(resp == "y" || resp == "Y")
			return true;
		else if(resp == "n" || resp == "N" || resp.empty())
			return false;
		std::cout<<"Please awnser with either \"y\" or \"n\"";
	}
}

int main(int argc, char** argv)
{
	Config config;
	argp_parse(&argp, argc, argv, 0, 0, &config);

	svm_set_print_string_function(&svmPrintFunction);

	if(config.networkFileName.empty() && (config.mode == MODE_ANN || config.mode == MODE_SVM) && !config.show)
	{
		Log(Log::ERROR)<<"A network file name must be specified";
		return -1;
	}

	if(config.spectraFileName.empty() && config.fileType != "trash")
	{
		Log(Log::ERROR)<<"A spectrum must be supplied";
		return -1;
	}

	torch::Tensor input;

	std::vector<eis::DataPoint> data;
	if(config.fileType == "csv")
	{
		data = loaders::csv(config.spectraFileName);
	}
	else if(config.fileType == "mpt")
	{
		data = loaders::mpt(config.spectraFileName);
	}
	else if(config.fileType == "trash")
	{
		TrashDataset trashSet;
		int seed;
		try
		{
			seed = config.spectraFileName.empty() ? 0 : std::stoi(config.spectraFileName.c_str());
		}
		catch(std::invalid_argument& ex)
		{
			seed = 0;
		}
		torch::data::Example<torch::Tensor, torch::Tensor> example = trashSet.get(seed);
		input = example.data;
	}
	else
	{
		Log(Log::ERROR)<<config.fileType<<" is not a valid file type";
		return -1;
	}

	if(data.empty() && !input.numel())
		return -1;

	EisGeneratorDataset::filterData(data, DATA_WIDTH);

	if(!input.numel())
		input = eisToTorch(data);

	if(config.show)
	{
		if(data.empty())
			data = torchToEis(input);
		printDataVect(data);
		return 0;
	}

	std::unique_ptr<EisGeneratorDataset> dataset (EisGeneratorDataset::getNeisGeneratedDataset(DATA_WIDTH));

	torch::Tensor output;

	if(!config.svmPre.empty())
	{
		if(!svm::filter(input, config.svmPre))
		{
			Log(Log::WARN)<<"input not recognized as EIS spectra";
			if(!yesNoPrompt("Try and classify anyways?"))
				return 0;
		}
	}

	switch(config.mode)
	{
		case MODE_ANN:
			output = ann::use(input, config.networkFileName);
			break;
		case MODE_NN:
			output = NearestNeighbor::use(input, dataset.get());
			break;
		case MODE_SVM:
			output = svm::use(input, config.networkFileName);
			break;
		case MODE_KNN:
		default:
			Log(Log::ERROR)<<"Not implemented";
			return -1;
	}

	output = output.reshape({output.numel()});

	Log(Log::INFO)<<std::setprecision(3)<<std::fixed<<std::setw(3)<<"Classes:\nNumber\tModel String\tLikelyhood";
	for(size_t i = 0; i < dataset->modelCount(); ++i)
	{
		std::string model = dataset->getModelStrFromClass(i);
		Log(Log::INFO)<<i<<'\t'<<model<<(model.size() >= 8 ? "\t" : "\t\t")<<output[i].item().to<float>();
	}

	for(int64_t i = 0; i < output.size(0); ++i)
	{
		if(output[i].item().to<float>() > 0.25)
			std::cout<<"Could be: "<<dataset->getModelStrFromClass(i)<<" confidence "<<output[i].item().to<float>()<<std::endl;
	}

	return 0;
}
