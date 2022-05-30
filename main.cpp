#include <cstddef>
#include <iostream>
#include <libsvm/svm.h>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include <eisgenerator/eistype.h>
#include <eisgenerator/basicmath.h>
#include <eisgenerator/normalize.h>

#include "net.h"
#include "log.h"
#include "options.h"
#include "csv.h"
#include "tokenize.h"
#include "eistotorch.h"
#include "eisgendata.h"
#include "nearest.h"
#include "svmloader.h"
#include "trashgenerator.h"

#include "constants.h"

void printDataVect(const std::vector<eis::DataPoint>& in)
{
	std::cout<<"omega, real, imag\n";
	for(const eis::DataPoint& res : in)
			std::cout<<res.omega<<','<<res.im.real()<<','<<res.im.imag()<<'\n';
}

std::vector<eis::DataPoint> parsEisSpectraFromMpt(std::string fileName)
{
	try
	{
		io::CSVReader<3, io::trim_chars<' ', '\t'>, io::no_quote_escape<'	'>> spectrumCsv(fileName);

		char* line = spectrumCsv.next_line();
		if(std::string(line) != "EC-Lab ASCII FILE")
		{
			Log(Log::ERROR)<<fileName<<" is not a valid EC-Lab ASCII file";
			return std::vector<eis::DataPoint>();
		}
		while((line = spectrumCsv.next_line()))
		{
			std::string lineStrr(line);
			if(lineStrr.find("Nb header lines") != std::string::npos)
			{
				std::vector<std::string> tokens = tokenize(lineStrr, ':');
				if(tokens.size() != 2)
					break;

				int header_lines = std::stoi(tokens[1]);
				int skiped_lines = spectrumCsv.get_file_line()+1;
				Log(Log::DEBUG)<<"Header Lines: "<<header_lines;
				for(int i = 0; i < header_lines-skiped_lines; ++i)
					 spectrumCsv.next_line();
				break;
			}
		}

		spectrumCsv.read_header(io::ignore_extra_column, "freq/Hz", "Re(Z)/Ohm", "-Im(Z)/Ohm");
		std::vector<eis::DataPoint> data;

		double hertz, real, imag;
		while(spectrumCsv.read_row(hertz, real, imag))
		{
			eis::DataPoint point;
			point.omega = hertz*2*M_PI;
			point.im.real(real);
			point.im.imag(0-imag);
			data.push_back(point);
		}

		if(data.size() > 1 && data.begin()->omega > (data.end()-1)->omega)
			std::reverse(data.begin(), data.end());
		return data;
	}
	catch(io::error::base& error)
	{
		Log(Log::ERROR)<<error.what();
	}
	return std::vector<eis::DataPoint>();
}

std::vector<eis::DataPoint> parsEisSpectraFromCsv(std::string fileName)
{
	try
	{
		io::CSVReader<3, io::trim_chars<' ', '\t'>, io::no_quote_escape<','>> spectrumCsv(fileName);
		spectrumCsv.read_header(io::ignore_extra_column, "omega", "real", "im");
		std::vector<eis::DataPoint> data;

		double omega, real, imag;
		while(spectrumCsv.read_row(omega, real, imag))
		{
			eis::DataPoint point;
			point.omega = omega;
			point.im.real(real);
			point.im.imag(imag);
			data.push_back(point);
		}
		return data;
	}
	catch(io::error::base& error)
	{
		Log(Log::ERROR)<<error.what();
	}
	return std::vector<eis::DataPoint>();
}

torch::Tensor useSvm(torch::Tensor input, std::string filenName)
{
	svm_model* model = svm_load_model(filenName.c_str());
	if(!model)
	{
		Log(Log::ERROR)<<"could not load SVM model from "<<filenName;
		exit(-1);
	}

	svm_node* data = SvmLoader<EisGeneratorDataset>::tensorToNewSvmNodeArray(input);

	Log(Log::DEBUG)<<"svm type: "<<svm_get_svm_type(model);
	if(svm_get_svm_type(model) != C_SVC)
	{
		Log(Log::ERROR)<<"Svm model "<<filenName<<" must be of type "<<C_SVC<<" but is of type "<<svm_get_svm_type(model);
		exit(-1);
	}

	double* estimates = new double[svm_get_nr_class(model)];
	svm_predict_probability(model, data, estimates);

	torch::Tensor output = torch::empty(svm_get_nr_class(model));

	for(int64_t i = 0; i < output.size(0); ++i)
		output[i] = estimates[i];

	svm_free_and_destroy_model(&model);
	delete[] data;
	delete[] estimates;
	return output;
}


bool svmFilter(torch::Tensor input, std::string filenName)
{
	svm_model* model = svm_load_model(filenName.c_str());
	if(!model)
	{
		Log(Log::ERROR)<<"could not load SVM model from "<<filenName;
		exit(-1);
	}

	svm_node* data = SvmLoader<EisGeneratorDataset>::tensorToNewSvmNodeArray(input);

	Log(Log::DEBUG)<<"svm type: "<<svm_get_svm_type(model);
	if(svm_get_svm_type(model) != ONE_CLASS)
	{
		Log(Log::ERROR)<<"Svm model "<<filenName<<" must be of type "<<ONE_CLASS<<" but is of type "<<svm_get_svm_type(model);
		exit(-1);
	}

	double prediction = svm_predict(model, data);
	Log(Log::DEBUG)<<"filter prediction: "<<prediction;
	svm_free_and_destroy_model(&model);
	delete[] data;
	return prediction;
}

torch::Tensor useAnn(torch::Tensor input, std::string fileName)
{
	std::shared_ptr<Net> net(new Net);
	net->eval();
	torch::load(net, fileName);
	torch::Tensor output = torch::exp(net->forward(input.reshape({1, net->getInputSize()})));
	return output;
}

torch::Tensor useNn(torch::Tensor input, EisGeneratorDataset* dataset)
{
	input = input.reshape({1, input.size(0)});
	torch::data::DataLoaderOptions options;
	options = options.batch_size(dataset->size().value()).workers(16);
	auto dataLoader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(dataset->map(torch::data::transforms::Stack<>()), options);
	torch::data::Example<torch::Tensor, torch::Tensor> grandBatch = *dataLoader->begin();
	NearestNeighbor nearest(grandBatch.data, grandBatch.target);
	size_t classNum = nearest.predict(input)[0].item().to<float>();
	torch::Tensor output = torch::zeros(dataset->modelCount());
	output[classNum] = 1.0;
	return output;
}

bool yesNowPrompt(std::string msg)
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
		data = parsEisSpectraFromCsv(config.spectraFileName);
	}
	else if(config.fileType == "mpt")
	{
		data = parsEisSpectraFromMpt(config.spectraFileName);
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
		if(!data.empty())
			printDataVect(data);
		else
			std::cout<<input<<'\n';
		return 0;
	}

	std::unique_ptr<EisGeneratorDataset> dataset (EisGeneratorDataset::getNeisGeneratedDataset(DATA_WIDTH));

	torch::Tensor output;

	if(!config.svmPre.empty())
	{
		if(!svmFilter(input, config.svmPre))
		{
			Log(Log::WARN)<<"input not recognized as EIS spectra";
			if(!yesNowPrompt("Try and classify anyways?"))
				return 0;
		}
	}

	switch(config.mode)
	{
		case MODE_ANN:
			output = useAnn(input, config.networkFileName);
			break;
		case MODE_NN:
			output = useNn(input, dataset.get());
			break;
		case MODE_SVM:
			output = useSvm(input, config.networkFileName);
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
