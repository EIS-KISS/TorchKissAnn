#include "csv.h"
#include "../csvBackend.h"
#include "../../utils/log.h"

using namespace loaders;

std::vector<eis::DataPoint> loaders::csv(std::string fileName)
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
