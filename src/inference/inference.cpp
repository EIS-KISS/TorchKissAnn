#include <cstddef>
#include <cstdio>
#include <eisgenerator/eistype.h>
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include <eisgenerator/model.h>

#include "tokenize.h"
#include "utils/log.h"
#include "options.h"
#include "data/loaders/mpt.h"
#include "data/generators/trashgenerator.h"
#include "data/loaders/dirloader.h"
#include "data/eistotorch.h"
#include "data/generators/eisgendata.h"
#include "nearest/nearest.h"
#include "utils/log.h"
#include "ann/classification.h"
#include "ann/regression.h"
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
		case FILE_TYPE_GENERATE:
			try
			{
				Log(Log::DEBUG)<<"Data width: "<<DATA_WIDTH;
				eis::Model model(fileName, 0, false);
				Log(Log::INFO)<<"Generateing spectra for "<<model.getModelStrWithParam();
				if(model.isParamSweep())
				{
					Log(Log::WARN)<<fileName<<" is a sweep. "
						<<"This is not supported, only the first element of the sweep will be used";
				}
				eis::Range omega(1, 1e6, DATA_WIDTH/2, true);
				return model.executeSweep(omega);
			}
			catch(const eis::parse_errror& err)
			{
				Log(Log::ERROR)<<"Could not load parse "<<fileName<<": "<<err.what();
				return std::vector<eis::DataPoint>();
			}
		case FILE_TYPE_CSV:
			try
			{
				return eis::loadFromDisk(fileName).data;
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

torch::Tensor testOnDir(const std::string& datasetDir, std::shared_ptr<ann::Net> network)
{
	EisDirDataset dataset(datasetDir);
	torch::data::DataLoaderOptions options;
	options = options.batch_size(BATCH_SIZE).workers(JOBS);
	auto dataLoader = torch::data::make_data_loader(dataset.map(torch::data::transforms::Stack<>()), options);
	return ann::classification::test(network, *dataLoader, dataset.size().value(),
				   dataset.classesCount(), dataset.classWeights(), dataset.isMulticlass());
}

torch::Tensor classify(torch::Tensor& input, PredictionMode mode, std::shared_ptr<ann::Net> net)
{
	torch::Tensor output;

	switch(mode)
	{
		case MODE_ANN:
			output = ann::classification::use(input, net);
			break;
		case MODE_NN:
		{
			std::unique_ptr<EisGeneratorDataset> dataset (EisGeneratorDataset::newDefaultDataset(DATA_WIDTH, 1e6));
			output = NearestNeighbor::use(input, dataset.get());
			break;
		}
		default:
			Log(Log::ERROR)<<"Not implemented";
			return torch::Tensor();
	}
	output = output.reshape({output.numel()});
	return output;
}

int testPipe(const Config& config, std::shared_ptr<ann::Net> net)
{
	if(config.datasetDir.empty())
	{
		Log(Log::ERROR)<<"You must specify a dataset to test on\n";
		return 1;
	}

	testOnDir(config.datasetDir, net);

	return 0;
}

int classifyPipeline(const Config& config, std::shared_ptr<ann::Net> net)
{
	std::vector<eis::DataPoint> data = grabData(config.spectraFileName, config.fileType);

	if(data.empty())
		return 3;

	filterData(data, DATA_WIDTH);

	torch::Tensor input = eisToTorch(data);

	if(!filter(input, config.filterFileName, config.filterMode))
		return 4;

	torch::Tensor output = classify(input, config.mode, net);

	if(output.numel() == 0)
		return 5;

	Log(Log::INFO)<<std::setprecision(3)<<std::fixed<<std::setw(3)<<"Classes:\nNumber\tModel String\tLikelyhood";

	for(int64_t i = 0; i < output.size(0); ++i)
	{
		std::string model;
		if(i < static_cast<int64_t>(net->getOutputLabels().size()))
			model = net->getOutputLabels()[i];
		else
			model = std::string("Unkown");
		Log(Log::INFO)<<i<<'\t'<<model<<(model.size() >= 8 ? "\t" : "\t\t")<<output[i].item().to<float>();
	}

	for(int64_t i = 0; i < output.size(0); ++i)
	{
		if(output[i].item().to<float>() > 0.15)
		{
			std::string model;
			if(i < static_cast<int64_t>(net->getOutputLabels().size()))
				model = net->getOutputLabels()[i];
			else
				model = std::to_string(i);
			std::cout<<"Could be: "<<model<<" confidence "<<output[i].item().to<float>()<<std::endl;
		}
	}

	return 0;
}

int regressionPipe(const Config& config, std::shared_ptr<ann::Net> net)
{
	std::vector<eis::DataPoint> data = grabData(config.spectraFileName, config.fileType);

	if(data.empty())
		return 3;

	std::vector<std::string> purposeTokens = tokenize(net->getPurpose(), ' ');
	if(purposeTokens[0] != "Regression")
	{
		Log(Log::ERROR)<<"The loaded model dosent report to be a regression model";
		return 1;
	}

	std::string targetModelStr = "Unkown";
	if(purposeTokens.size() >= 3)
		targetModelStr = purposeTokens[2];

	Log(Log::INFO)<<"Running parameter regression for "<<targetModelStr;

	torch::Tensor input = eisToTorch(data);

	Log(Log::DEBUG)<<"data size: "<<data.size()<<" tensor size "<<input.sizes();

	torch::Tensor output = ann::regression::use(input, net);

	Log(Log::DEBUG)<<"output shape: "<<output.sizes();

	if(output.numel() == 0)
		return 5;

	for(int64_t i = 0; i < output.size(0); ++i)
	{
		std::string parameter;
		if(i < static_cast<int64_t>(net->getOutputLabels().size()))
			parameter = net->getOutputLabels()[i];
		else
			parameter = std::string("Unkown");
		Log(Log::INFO)<<i<<'\t'<<parameter<<(parameter.size() >= 8 ? "\t" : "\t\t")<<output[i].item().to<float>();
	}

	return 0;
}

int showPipe(const Config& config)
{
	std::vector<eis::DataPoint> data = grabData(config.spectraFileName, config.fileType);
	if(data.empty())
		return 3;
	printDataVect(data);
	return 0;
}

int reexportPipe(const Config& config)
{
	std::vector<eis::DataPoint> data = grabData(config.spectraFileName, config.fileType);
	if(data.empty())
		return 3;
	filterData(data, DATA_WIDTH);
	torch::Tensor input = eisToTorch(data);
	eis::saveToDisk(eis::EisSpectra(torchToEis(input), "", ""), "export.txt");
	return 0;
}

bool requiresNetwork(PredictionMode mode)
{
	switch(mode)
	{
		case MODE_NN:
		case MODE_ANN:
		case MODE_ANN_TEST:
		case MODE_REGRESSION:
			return true;
		case MODE_SHOW:
		case MODE_REEXPORT:
		case MODE_INVALID:
		default:
			return false;
	}
}

int main(int argc, char** argv)
{
	Config config;
	argp_parse(&argp, argc, argv, 0, 0, &config);

	std::shared_ptr<ann::Net> net;

	if(requiresNetwork(config.mode))
	{
		if(config.networkFileName.empty())
		{
			Log(Log::ERROR)<<"A network filename is required for this mode";
			return 2;
		}

		net = ann::Net::newNetFromCheckpointDir(config.networkFileName);

		if(!net)
		{
			Log(Log::ERROR)<<"Could not load network from "<<config.networkFileName;
			return 2;
		}

		Log(Log::INFO)<<"Loaded network with "<<net->getOutputSize()<<" outputs. Purpose: "<<net->getPurpose();
		const std::vector<std::string>& outputLabels = net->getOutputLabels();
		if(!outputLabels.empty())
		{
			Log(Log::INFO)<<"Output labels:";
			for(const std::string& label : outputLabels)
				Log(Log::INFO)<<label;
		}
	}

	switch(config.mode)
	{
		case MODE_NN:
		case MODE_ANN:
			return classifyPipeline(config, net);
		case MODE_ANN_TEST:
			return testPipe(config, net);
		case MODE_REGRESSION:
			return regressionPipe(config, net);
		case MODE_SHOW:
			return showPipe(config);
		case MODE_REEXPORT:
			return reexportPipe(config);
		case MODE_INVALID:
		default:
			Log(Log::ERROR)<<"An invalid mode was specified";
			return 1;
	}

	return 0;
}
