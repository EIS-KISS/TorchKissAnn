#include "mpt.h"

#include <vector>
#include <string>
#include <eisgenerator/eistype.h>

#include "../csvBackend.h"
#include "../../utils/log.h"
#include "../../utils/tokenize.h"

std::vector<eis::DataPoint> loaders::mpt(std::string fileName)
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
