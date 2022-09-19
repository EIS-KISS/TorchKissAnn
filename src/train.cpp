#include <cstddef>
#include <cstdint>
#include <ios>
#include <iostream>
#include <eisgenerator/model.h>
#include <memory>
#include <vector>

#include "ann/ann.h"
#include "svm/svm.h"
#include "gan/gan.h"
#include "log.h"
#include "data/print.h"
#include "data/generators/eisgendata.h"
#include "data/generators/deduplicator.h"
#include "trainoptions.h"

int main(int argc, char** argv)
{
	std::cout<<std::setprecision(5)<<std::fixed<<std::setw(3);
	Log::level = Log::INFO;

	Config config;
	argp_parse(&argp, argc, argv, 0, 0, &config);

	if(config.mode == MODE_INVALID)
	{
		Log(Log::ERROR)<<"You must specify what to train: -m ann, svm or svmcross";
		return -1;
	}

	svm_set_print_string_function(&svmPrintFunction);

	std::vector<size_t> skiparg;

	switch(config.mode)
	{
		case MODE_ANN:
			ann::train("net.vos", EisGeneratorDataset::DEFAULT_EXAMPLE_COUNT, config.noise, false);
			break;
		case MODE_ANN_CONV:
			ann::train("netConv.vos", EisGeneratorDataset::DEFAULT_EXAMPLE_COUNT, config.noise, true);
			break;
		case MODE_SVM:
			svm::train(false);
			break;
		case MODE_SVM_FILTER:
			svm::train(true);
			break;
		case MODE_SVM_CROSS:
			svm::cross();
			break;
		case MODE_GAN:
			gan::train("ganNet", EisGeneratorDataset::DEFAULT_EXAMPLE_COUNT, config.noise);
			break;
		case MODE_DEDUP: //350GiB of ram required!!!
		{
			EisGeneratorDataset *dataset =
				EisGeneratorDataset::getNeisGeneratedDataset(DATA_WIDTH, EisGeneratorDataset::DEFAULT_EXAMPLE_COUNT, 0, false, true);
			DeduplatorDataset dedupData(dataset, dataset->classesCount());
			dedupData.save("dedupInfo.csv");
		}
		break;
		case MODE_INVALID:
		default:
			Log(Log::ERROR)<<"invalid model";
			break;
	}

	return 0;
}
