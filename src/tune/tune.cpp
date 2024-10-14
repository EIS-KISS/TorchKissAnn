#include <eisgenerator/log.h>

#include "ann/classification.h"
#include "ann/regression.h"
#include "gan/gan.h"
#include "log.h"
#include "options.h"
#include "globals.h"

int main(int argc, char** argv)
{
	std::cout<<std::setprecision(5)<<std::fixed<<std::setw(3);
	Log::level = Log::INFO;
	eis::Log::level = eis::Log::ERROR;

	Config config;
	argp_parse(&argp, argc, argv, 0, 0, &config);

	choose_device(config.noGpu);

	if(config.mode == MODE_INVALID)
	{
		Log(Log::ERROR)<<"You must specify what to train: -m " MODE_LIST;
		return 3;
	}

	switch (config.mode)
	{
		case MODE_ANN:
		case MODE_ANN_CONV:
		case MODE_GAN:
			Log(Log::ERROR)<<"Tuneing is not yet supported in the mode "<<trainModeToStr(config.mode);
			break;

		case MODE_REGRESSION:
			//ann::regression::parameterSearch();
			//break;
		default:
			Log(Log::ERROR)<<"You must specify what to train: -m " MODE_LIST;
			return 3;
	}

	free_device();
	return 0;
}
