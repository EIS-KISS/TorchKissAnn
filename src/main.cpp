#include <cstddef>
#include <cstdio>
#include <eisgenerator/eistype.h>
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>

#include "utils/log.h"
#include "options.h"
#include "data/loaders/mpt.h"
#include "data/generators/trashgenerator.h"
#include "data/eistotorch.h"
#include "data/generators/eisgendata.h"
#include "nearest/nearest.h"
#include "utils/log.h"
#include "ann/ann.h"
#include "gan/gan.h"
#include "data/print.h"
#include "data/filterdata.h"
#include "constants.h"

bool yesNoPrompt(std::string msg)
{
	while(true)
	{
		std::cout<<msg<<" [y/n] ";
		std::string resp;
		std::cin>>resp;
		if(resp == "y" || resp == "Y")
			return true;
		else if(resp == "n" || resp == "N" || resp.empty())
			return false;
		std::cout<<"Please awnser with either \"y\" or \"n\"\n";
	}
}

std::vector<eis::DataPoint> grabData(const std::string& fileName, FileType type)
{
	if(fileName.empty() && type != FILE_TYPE_TRASH)
	{
		Log(Log::ERROR)<<"A spectrum must be supplied";
		return std::vector<eis::DataPoint>();
	}

	switch(type)
	{
		case FILE_TYPE_CSV:
			try
			{
				return eis::loadFromDisk(fileName).first;
			}
			catch(const eis::file_error& ex)
			{
				Log(Log::ERROR)<<"Could not load file(s): "<<ex.what();
				return std::vector<eis::DataPoint>();
			}
		case FILE_TYPE_MPT:
			return loaders::mpt(fileName);
		case FILE_TYPE_TRASH:
		{
			TrashDataset trashSet;
			int seed;
			try
			{
				seed = fileName.empty() ? 0 : std::stoi(fileName.c_str());
			}
			catch(std::invalid_argument& ex)
			{
				seed = 0;
			}
			torch::data::Example<torch::Tensor, torch::Tensor> example = trashSet.get(seed);
			return torchToEis(example.data);
		}
		case FILE_TYPE_GAN:
		{
			torch::Tensor data = gan::generate(fileName);
			return torchToEis(data);
		}
		case FILE_TYPE_RELAXIS:
		case FILE_TYPE_INVALID:
		default:
			Log(Log::ERROR)<<type<<" is not a valid file type";
			return std::vector<eis::DataPoint>();
	}
}

bool filter(torch::Tensor& data, const std::string& fileName, FilterMode filterMode)
{
	if(filterMode != FILTER_NONE && fileName.empty())
	{
		Log(Log::ERROR)<<"A filter network file is required for this filter method";
		return false;
	}

	bool pass;
	switch(filterMode)
	{
		case FILTER_NONE:
			return true;
		case FILTER_GAN:
			pass = gan::filter(data, fileName);
			break;
		case FILTER_INVALID:
		default:
			Log(Log::ERROR)<<"no valid filter";
			return false;
	}

	if(!pass)
	{
		Log(Log::WARN)<<"input not recognized as EIS spectra";
		pass = yesNoPrompt("Try and classify anyways?");
		if(pass)
			Log(Log::WARN)<<"Classification will be unreliable!!!";
	}

	return pass;
}

torch::Tensor classify(torch::Tensor& input, PredictionMode mode, std::string networkFileName,
					   size_t outputSize, size_t extraLayers, const std::string exportDir, double exportProb)
{
	torch::Tensor output;

	if(networkFileName.empty() && (mode == MODE_ANN))
	{
		Log(Log::ERROR)<<"A network file name must be specified";
		return torch::Tensor();
	}

	switch(mode)
	{
		case MODE_ANN:
			output = ann::use(input, ann::load(networkFileName, DATA_WIDTH, outputSize, false, 4, extraLayers));
			break;
		case MODE_ANN_TEST:
			output = ann::testOnGenNeis(ann::load(networkFileName, DATA_WIDTH, outputSize, false, 4, extraLayers), exportDir, exportProb);
			break;
		case MODE_NN:
		{
			std::unique_ptr<EisGeneratorDataset> dataset (EisGeneratorDataset::newDefaultDataset(DATA_WIDTH, 1e6));
			output = NearestNeighbor::use(input, dataset.get());
			break;
		}
		case MODE_KNN:
		default:
			Log(Log::ERROR)<<"Not implemented";
			return torch::Tensor();
	}
	output = output.reshape({output.numel()});
	return output;
}

int main(int argc, char** argv)
{
	Config config;
	argp_parse(&argp, argc, argv, 0, 0, &config);

	std::vector<eis::DataPoint> data;

	if(config.mode != MODE_ANN_TEST)
	{
		data = grabData(config.spectraFileName, config.fileType);

		if(data.empty())
		{
			Log(Log::ERROR)<<"Could not load spectrum file from "<<config.spectraFileName;
			return 3;
		}

		filterData(data, DATA_WIDTH);

		if(config.show)
		{
			printDataVect(data);
			return 0;
		}

		if(config.reexport)
		{
			torch::Tensor input = eisToTorch(data);
			eis::saveToDisk(torchToEis(input), "export.txt");
			return 0;
		}
	}

	torch::Tensor input = eisToTorch(data);

	if(!filter(input, config.filterFileName, config.filterMode))
		return 4;

	torch::Tensor output = classify(input, config.mode, config.networkFileName, config.outputSize,
									config.extraLayers, config.exportDir, config.outputProb);

	if(output.numel() == 0)
		return 5;

	std::unique_ptr<EisGeneratorDataset> dataset (EisGeneratorDataset::newDefaultDataset(DATA_WIDTH, 1e6));

	Log(Log::INFO)<<std::setprecision(3)<<std::fixed<<std::setw(3)<<"Classes:\nNumber\tModel String\tLikelyhood";

	for(int64_t i = 0; i < std::min(static_cast<int64_t>(dataset->classesCount()), output.size(0)); ++i)
	{
		std::string model;
		if(static_cast<int64_t>(dataset->classesCount()) == output.size(0))
			model = dataset->modelStringForClass(i);
		else
			model = std::string("Unkown");
		Log(Log::INFO)<<i<<'\t'<<model<<(model.size() >= 8 ? "\t" : "\t\t")<<output[i].item().to<float>();
	}

	for(int64_t i = 0; i < output.size(0); ++i)
	{
		if(output[i].item().to<float>() > 0.15)
		{
			std::string model;
			if(static_cast<int64_t>(dataset->classesCount()) == output.size(0))
				model = dataset->modelStringForClass(i);
			else
				model = std::to_string(i);
			std::cout<<"Could be: "<<model<<" confidence "<<output[i].item().to<float>()<<std::endl;
		}
	}

	return 0;
}
