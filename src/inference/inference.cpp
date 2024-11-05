#include <cstdio>
#include <eisgenerator/basicmath.h>
#include <kisstype/type.h>
#include <kisstype/spectra.h>
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include <eisgenerator/model.h>

#include "commonoptions.h"
#include "tokenize.h"
#include "utils/log.h"
#include "options.h"
#include "data/loaders/dirloader.h"
#include "data/eistotorch.h"
#include "utils/log.h"
#include "ann/classification.h"
#include "ann/regression.h"
#include "gan/gan.h"
#include "data/print.h"
#include "globals.h"
#include <fstream>

static bool yesNoPrompt(std::string msg)
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

static std::pair<std::vector<eis::DataPoint>, std::vector<fvalue>> grabData(const std::string& fileName, FileType type, const std::vector<std::pair<std::string, int64_t>>& extraInputs)
{
	if(fileName.empty())
	{
		Log(Log::ERROR)<<"A spectrum must be supplied";
		return {std::vector<eis::DataPoint>(), std::vector<fvalue>()};
	}

	if(!extraInputs.empty() && type != FILE_TYPE_CSV)
	{
		Log(Log::ERROR)<<"This network requires extra inputs besides EIS spectra, but only the csv file type supports extra inputs";
		return {std::vector<eis::DataPoint>(), std::vector<fvalue>()};
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
				return {model.executeSweep(omega), {}};
			}
			catch(const eis::parse_errror& err)
			{
				Log(Log::ERROR)<<"Could not load parse "<<fileName<<": "<<err.what();
				return {{}, {}};
			}
		case FILE_TYPE_CSV:
			try
			{
				eis::Spectra spectra = eis::Spectra::loadFromDisk(fileName);
				std::vector<eis::DataPoint> data = spectra.data;
				std::vector<fvalue> extra;

				for(const std::pair<std::string, int64_t>& input : extraInputs)
				{
					if(input.second != 1)
					{
						Log(Log::ERROR)<<"This network requires the extra input \""<<input.first
							<<"\" with a length of "<<input.second<<" but this type of file can only provide extra inputs of length 1";
						return {{}, {}};
					}
					auto search = std::find(spectra.labelNames.begin(), spectra.labelNames.end(), input.first);
					if(search == spectra.labelNames.end())
					{
						Log(Log::ERROR)<<"This network requires the extra input \""<<input.first<<"\" but the provided csv file dose not contain this";
						return {{}, {}};
					}
					extra.push_back(static_cast<fvalue>(spectra.labels[search-spectra.labelNames.begin()]));
				}
				return {data, extra};
			}
			catch(const eis::file_error& ex)
			{
				Log(Log::ERROR)<<"Could not load file(s): "<<ex.what();
				return {{}, {}};
			}
		case FILE_TYPE_GAN:
		{
			torch::Tensor data = gan::generate(fileName);
			return {torchToEis(data), {}};
		}
		case FILE_TYPE_RELAXIS:
		case FILE_TYPE_INVALID:
		default:
			Log(Log::ERROR)<<type<<" is not a valid file type";
			return {{}, {}};
	}
}

static bool filter(torch::Tensor& data, const std::string& fileName, FilterMode filterMode)
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

static torch::Tensor classify(torch::Tensor& input, PredictionMode mode, std::shared_ptr<ann::Net> net)
{
	torch::Tensor output;

	switch(mode)
	{
		case MODE_ANN:
			output = ann::classification::use(input, net);
			break;
		default:
			Log(Log::ERROR)<<"Not implemented";
			return torch::Tensor();
	}
	output = output.reshape({output.numel()});
	return output;
}

static int classifyPipeline(const Config& config, std::shared_ptr<ann::Net> net)
{
	std::vector<eis::DataPoint> data = grabData(config.spectraFileName, config.fileType, net->getExtraInputs()).first;

	if(net->getPurpose() != "Classifier")
	{
		Log(Log::ERROR)<<"The loaded model dosent report to be a classifier";
		return 1;
	}

	if(data.empty())
		return 3;

	torch::Tensor input = eisToTorch(data);

	if(!filter(input, config.filterFileName, config.filterMode))
		return 4;

	Log(Log::DEBUG)<<"Input data:\n"<<input;

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

static int regressionPipe(const Config& config, std::shared_ptr<ann::Net> net)
{
	std::pair<std::vector<eis::DataPoint>, std::vector<fvalue>> data = grabData(config.spectraFileName, config.fileType, net->getExtraInputs());

	if(data.first.empty())
		return 3;

	std::vector<std::string> purposeTokens = tokenize(net->getPurpose(), ',');
	if(purposeTokens[0] != "Regression")
	{
		Log(Log::ERROR)<<"The loaded model dosent report to be a regression model";
		return 1;
	}

	std::string targetModelStr = "Unkown";
	if(purposeTokens.size() >= 3)
		targetModelStr = purposeTokens[1];

	Log(Log::INFO)<<"Running parameter regression for "<<targetModelStr;

	Log(Log::DEBUG)<<"data size: "<<data.first.size();

	if(static_cast<int64_t>(data.first.size()*2 + data.second.size()) != net->getInputSize())
	{
		Log(Log::INFO)<<"Network trained for an input size of "<<net->getInputSize()
		<<" supplied data has a size of "<<data.first.size()*2<<" and will be rescaled";

		data.first = eis::rescale(data.first, net->getInputSize()/2 - data.second.size());
	}
	torch::Tensor input = eisToTorchExtra(data.first, data.second).to(*offload_device);
	Log(Log::DEBUG)<<"input: "<<input;

	torch::Tensor output = ann::regression::use(input, net).to(torch::kCPU);

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

static int showPipe(const Config& config)
{
	std::vector<eis::DataPoint> data = grabData(config.spectraFileName, config.fileType, {}).first;
	if(data.empty())
		return 3;
	printDataVect(data);
	return 0;
}

static int reexportPipe(const Config& config)
{
	std::vector<eis::DataPoint> data = grabData(config.spectraFileName, config.fileType, {}).first;
	if(data.empty())
		return 3;
	torch::Tensor input = eisToTorch(data);
	eis::Spectra(torchToEis(input), "", "").saveToDisk("export.txt");
	return 0;
}

static bool requiresNetwork(PredictionMode mode)
{
	switch(mode)
	{
		case MODE_ANN:
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
	Log::level = Log::INFO;
	argp_parse(&argp, argc, argv, 0, 0, &config);

	choose_device(true);

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

		net->to(*offload_device);

		Log(Log::INFO)<<"Loaded network with "<<net->getOutputSize()<<" outputs. Purpose: "<<net->getPurpose();
		const std::vector<std::string>& outputLabels = net->getOutputLabels();
		if(!outputLabels.empty())
		{
			Log(Log::INFO)<<"Output labels:";
			for(const std::string& label : outputLabels)
				Log(Log::INFO)<<label;
		}

		net->eval();
	}

	switch(config.mode)
	{
		case MODE_ANN:
			return classifyPipeline(config, net);
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

	free_device();
	return 0;
}
