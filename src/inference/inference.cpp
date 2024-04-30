#include <cstddef>
#include <cstdio>
#include <eisgenerator/basicmath.h>
#include <eisgenerator/eistype.h>
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include <eisgenerator/model.h>

#include "commonoptions.h"
#include "data/generators/eisgendatanoise.h"
#include "tokenize.h"
#include "utils/log.h"
#include "options.h"
#include "data/loaders/mpt.h"
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
#include "globals.h"
#include <fstream>

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
	if(fileName.empty())
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
				return eis::EisSpectra::loadFromDisk(fileName).data;
			}
			catch(const eis::file_error& ex)
			{
				Log(Log::ERROR)<<"Could not load file(s): "<<ex.what();
				return std::vector<eis::DataPoint>();
			}
		case FILE_TYPE_MPT:
			return loaders::mpt(fileName);
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


torch::Tensor testOnDir(const std::string& datasetDir, std::shared_ptr<ann::Net> network)
{
	EisDirDataset dataset(datasetDir);
	torch::data::DataLoaderOptions options;
	options = options.batch_size(batch_size).workers(JOBS);
	auto dataLoader = torch::data::make_data_loader(dataset.map(torch::data::transforms::Stack<>()), options);
	return ann::classification::test(network, *dataLoader, dataset.size().value(),
									 dataset.classesCount(), dataset.classWeights(), dataset.isMulticlass());
}

torch::Tensor testGen(const std::string& classesPath, std::shared_ptr<ann::Net> network)
{
	EisGeneratorDataset dataset(classesPath, 10000, network->getInputSize(), 0, false, false);
	torch::data::DataLoaderOptions options;
	options = options.batch_size(batch_size).workers(JOBS);
	auto dataLoader = torch::data::make_data_loader(dataset.map(torch::data::transforms::Stack<>()), options);
	return ann::classification::test(network, *dataLoader, dataset.size().value(),
									 dataset.classesCount(), dataset.classWeights(), dataset.isMulticlass());
}

int testPipe(const Config& config, std::shared_ptr<ann::Net> net)
{
	if(config.spectraFileName.empty())
	{
		Log(Log::ERROR)<<"You must specify a dataset to test on\n";
		return 1;
	}

	if(config.datasetMode == DATASET_DIR)
	{
		testOnDir(config.spectraFileName, net);
	}
	else if(config.datasetMode == DATASET_GEN)
	{
		testGen(config.spectraFileName, net);
	}
	else
	{
		Log(Log::ERROR)<<"Dataset not implemented for this operation: "<<datasetModeToStr(config.datasetMode);
		return 1;
	}
	return 0;
}

static torch::Tensor confusionOnDir(const std::string& datasetDir, std::shared_ptr<ann::Net> network)
{
	EisDirDataset dataset(datasetDir);
	torch::data::DataLoaderOptions options;
	options = options.batch_size(batch_size).workers(JOBS);
	auto dataLoader = torch::data::make_data_loader(dataset.map(torch::data::transforms::Stack<>()), options);
	return ann::classification::confusion(network, *dataLoader);
}

static torch::Tensor confusionOnGen(const std::string& classesPath, std::shared_ptr<ann::Net> network)
{
	EisGeneratorDataset dataset(classesPath, 10000, network->getInputSize(), 0, false, false);
	torch::data::DataLoaderOptions options;
	options = options.batch_size(batch_size).workers(JOBS);
	auto dataLoader = torch::data::make_data_loader(dataset.map(torch::data::transforms::Stack<>()), options);
	return ann::classification::confusion(network, *dataLoader);
}

static torch::Tensor confusionOnGenNoise(const std::string& classesPath, std::shared_ptr<ann::Net> network)
{
	EisGeneratorDatasetNoise dataset(classesPath, 10000, network->getInputSize(), false);
	torch::data::DataLoaderOptions options;
	options = options.batch_size(batch_size).workers(JOBS);
	auto dataLoader = torch::data::make_data_loader(dataset.map(torch::data::transforms::Stack<>()), options);
	return ann::classification::confusion(network, *dataLoader);
}

static int confusionPipe(const Config& config, std::shared_ptr<ann::Net> net)
{
	if(config.spectraFileName.empty())
	{
		Log(Log::ERROR)<<"You must specify a dataset to test on\n";
		return 1;
	}

	torch::Tensor confusionMatrix;

	if(config.datasetMode == DATASET_DIR)
	{
		confusionMatrix = confusionOnDir(config.spectraFileName, net);
	}
	else if(config.datasetMode == DATASET_GEN)
	{
		confusionMatrix = confusionOnGen(config.spectraFileName, net);
	}
	else if(config.datasetMode == DATASET_GEN_NOISE)
	{
		confusionMatrix = confusionOnGenNoise(config.spectraFileName, net);
	}
	else
	{
		Log(Log::ERROR)<<"Dataset not implemented for this operation: "<<datasetModeToStr(config.datasetMode);
		return 1;
	}

	std::string path = "./confusion.csv";
	std::ofstream file(path);
	if(!file.is_open())
	{
		Log(Log::ERROR)<<"Coult not open "<<path<<" for writeing";
		return 1;
	}

	file<<std::setw(10)<<",";
	for(int64_t i = 0; i < net->getOutputSize(); ++i)
		file<<net->getOutputLabels()[i]<<", ";
	file<<'\n';

	for(int64_t row = 0; row < net->getOutputSize(); ++row)
	{
		file<<net->getOutputLabels()[row]<<", ";
		for(int64_t col = 0; col < net->getOutputSize(); ++col)
			file<<confusionMatrix[row][col].item<long>()<<", ";
		file<<'\n';
	}

	Log(Log::INFO)<<"Saved confsion matrix to "<<path;

	return 0;
}

int classifyPipeline(const Config& config, std::shared_ptr<ann::Net> net)
{
	std::vector<eis::DataPoint> data = grabData(config.spectraFileName, config.fileType);

	if(net->getPurpose() != "Classifier")
	{
		Log(Log::ERROR)<<"The loaded model dosent report to be a classifier";
		return 1;
	}

	if(data.empty())
		return 3;

	filterData(data, net->getInputSize());

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

int regressionPipe(const Config& config, std::shared_ptr<ann::Net> net)
{
	std::vector<eis::DataPoint> data = grabData(config.spectraFileName, config.fileType);

	if(data.empty())
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

	Log(Log::DEBUG)<<"data size: "<<data.size();

	if(static_cast<int64_t>(data.size())*2 != net->getInputSize())
	{
		Log(Log::INFO)<<"Network trained for an input size of "<<net->getInputSize()
		<<" supplied data has a size of "<<data.size()*2<<" and will be rescaled";

		data = eis::rescale(data, net->getInputSize()/2);
	}
	torch::Tensor input = eisToTorch(data).to(*offload_device);
	Log(Log::DEBUG)<<"input tensor size: "<<input.numel();

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
	eis::EisSpectra(torchToEis(input), "", "").saveToDisk("export.txt");
	return 0;
}

bool requiresNetwork(PredictionMode mode)
{
	switch(mode)
	{
		case MODE_NN:
		case MODE_ANN:
		case MODE_ANN_TEST:
		case MODE_ANN_CONFUSION:
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
		case MODE_NN:
		case MODE_ANN:
			return classifyPipeline(config, net);
		case MODE_ANN_TEST:
			return testPipe(config, net);
		case MODE_ANN_CONFUSION:
			return confusionPipe(config, net);
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
