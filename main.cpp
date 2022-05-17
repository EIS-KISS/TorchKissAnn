#include <iostream>
#include <torch/torch.h>
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
#include "eisgeneratorloader.h"

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

int main(int argc, char** argv)
{
	Config config;
	argp_parse(&argp, argc, argv, 0, 0, &config);

	if(config.networkFileName.empty())
	{
		Log(Log::ERROR)<<"A network file name must be specified";
		return -1;
	}

	if(config.spectraFileName.empty())
	{
		Log(Log::ERROR)<<"A spectrum must be supplied";
		return -1;
	}

	std::vector<eis::DataPoint> data;
	if(config.fileType == "csv")
	{
		data = parsEisSpectraFromCsv(config.spectraFileName);
	}
	else if(config.fileType == "mpt")
	{
		data = parsEisSpectraFromMpt(config.spectraFileName);
	}
	else
	{
		Log(Log::ERROR)<<config.fileType<<" is not a valid file type";
		return -1;
	}

	std::shared_ptr<Net> net(new Net);
	torch::load(net, config.networkFileName);

	if(data.empty())
		return -1;

	EisGeneratorDataset::filterData(data, net->getInputSize());

	torch::Tensor input = eisToTorch(data);

	if(config.show)
	{
		printDataVect(data);
		return 0;
	}

	torch::Tensor output = net->forward(input.reshape({1, net->getInputSize()}));
	std::cout<<output<<'\n';
	std::cout<<output.argmax(1)[0]<<'\n';

	return 0;
}
