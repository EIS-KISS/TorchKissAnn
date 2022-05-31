#include <cstddef>
#include <ios>
#include <iostream>
#include <eisgenerator/model.h>
#include <memory>
#include <vector>

#include "ann/ann.h"
#include "svm/svm.h"
#include "log.h"
#include "data/print.h"

void printHelp(char* argo)
{
	std::cout<<"Usage: "<<argo<<" [OPTION...] [TARGET]\n";
	std::cout<<argo<<" training utility for torchkissann\n\n";
	std::cout<<"Options:\n";
	std::cout<<"-h, --help\tPrint this help\n";
	std::cout<<"-v, --verbose\tPrint training ifomation verbosly\n\n";
	std::cout<<"Targets, one or more of of:\n";
	std::cout<<"ann\n";
	std::cout<<"svm\n";
	std::cout<<"svmfilter\n";
	std::cout<<"svmcross\n";
}

int main(int argc, char** argv)
{
	std::cout<<std::setprecision(3)<<std::fixed<<std::setw(3);
	Log::level = Log::INFO;
	if(argc < 2)
	{
		Log(Log::ERROR)<<"You must specify what to train: ann, svm or svmcross";
		return -1;
	}

	svm_set_print_string_function(&svmPrintFunction);

	std::vector<size_t> skiparg;

	for(int i = 1; i < argc; ++i)
	{
		if(std::string(argv[i]) == "--verbose" || std::string(argv[i]) == "-v" )
		{
			skiparg.push_back(i);
			Log::level = Log::DEBUG;
		}
		else if(std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h")
		{
			printHelp(argv[0]);
			return 0;
		}
	}

	for(int i = 1; i < argc; ++i)
	{
		if(std::string(argv[i]) == "-v")
			Log::level = Log::DEBUG;
		else if(std::string(argv[i]) == "ann")
			ann::train();
		else if(std::string(argv[i]) == "svm")
			svm::train(false);
		else if(std::string(argv[i]) == "svmfilter")
			svm::train(true);
		else if(std::string(argv[i]) == "svmcross")
			svm::cross();
		else if(std::find(skiparg.begin(), skiparg.end(), i) != skiparg.end())
			Log(Log::ERROR)<<argv[i]<<" is not a supported trainable model";
	}

	return 0;
}
