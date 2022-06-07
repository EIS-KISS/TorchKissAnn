#include <cstddef>
#include <cstdint>
#include <ios>
#include <iostream>
#include <eisgenerator/model.h>
#include <memory>
#include <vector>

#include "ann/ann.h"
#include "svm/svm.h"
#include "log.h"
#include "data/print.h"
#include "trainoptions.h"

void printHelp(char* argo)
{
	std::cout<<"Usage: "<<argo<<" [OPTION...] [TARGET]\n";
	std::cout<<argo<<" training utility for torchkissann\n\n";
	std::cout<<"Options:\n";
	std::cout<<"-h, --help\tPrint this help\n";
	std::cout<<"-v, --verbose\tPrint training information verbosly\n\n";
	std::cout<<"Targets, one or more of of:\n";
	std::cout<<"ann\n";
	std::cout<<"svm\n";
	std::cout<<"svmfilter\n";
	std::cout<<"svmcross\n";
}

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
			ann::train();
			break;
		case MODE_SVM:
			svm::train(false);
			break;
		case MODE_SVM_FILTER:
			svm::train(true);
			break;
		case MODE_SVM_CROSS:
			svm::cross();
		case MODE_INVALID:
		default:
			Log(Log::ERROR)<<"invalid model";
			break;
	}

	return 0;
}
