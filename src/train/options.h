#pragma once
#include <string>
#include <vector>
#include <argp.h>
#include <iostream>
#include <filesystem>
#include "utils/log.h"
#include "commonoptions.h"

#define MODE_LIST "ann, conv, script, gan, regression, regression_script, autoencoder"

inline const char *argp_program_version = "TorchKissAnnTrain";
inline const char *argp_program_bug_address = "<carl@uvos.xyz>";
static char doc[] = "Application that trains models for TorchKissAnn";
static char args_doc[] = "";

static struct argp_option options[] =
{
  {"verbose",		'v', 0,				0,	"Show debug messages" },
  {"quiet", 		'q', 0,				0,	"only output data" },
  {"model", 		'm', "[STRING]",	0,	"model to train: " MODE_LIST},
  {"dataset", 		'd', "[STRING]",	0,	"dataset type to use for training: " DATASET_LIST},
  {"file", 			'f', "[STRING]",	0,	"filename for dataset"},
  {"test",			't', "[STRING]",	0,	"filename for the test dataset"},
  {"batch-size",	'b', "[NUMBER]",	0,	"size of the training batch"},
  {"epochs",		'i', "[NUMBER]",	0,	"maximum number of epochs to train, may exit earlier due to lack of progress"},
  {"extra-layers",	'l', "[NUMBER]",	0,	"extra hidden layers to use in ann and gan, default: 3"},
  {"learing-rate",	'r', "[NUMBER]",	0,	"sgd/adam lering rate, default: 0.005"},
  {"output-dir",	'o', "[DIRECTORY]", 0,	"directory where training logs and models will be saved"},
  {"cpu",			'c', 0,				0,	"don't use gpu even if one is available"},
  {"network",		'n', "[PATH]",		0,	"torchScript network to train"},
  {"no-weights",	'g', 0,				0, 	"Don't use class weights"},
  {"latent-size",	'a', "[NUMBER]",	0, 	"Size of the latent vector for the autoencoder"},
  { 0 }
};

typedef enum
{
	MODE_INVALID = -1,
	MODE_ANN = 0,
	MODE_ANN_CONV,
	MODE_ANN_SCRIPT,
	MODE_AUTO_SCRIPT,
	MODE_AUTO,
	MODE_GAN,
	MODE_REGRESSION,
	MODE_REGRESSION_SCRIPT
} TrainMode;

static std::string trainModeToStr(const TrainMode mode)
{
	switch(mode)
	{
		case MODE_ANN:
			return "ann";
		case MODE_ANN_CONV:
			return "conv";
		case MODE_ANN_SCRIPT:
			return "script";
		case MODE_AUTO_SCRIPT:
			return "autoencoder";
		case MODE_AUTO:
			return "autoencoder_simple";
		case MODE_GAN:
			return "gan";
		case MODE_REGRESSION:
			return "regression";
		case MODE_REGRESSION_SCRIPT:
			return "regression_script";
		default:
			return "Invalid";
	}
}

static TrainMode parseTrainMode(const std::string& in)
{
	if(in.empty() || in == trainModeToStr(MODE_ANN))
		return MODE_ANN;
	else if(in == trainModeToStr(MODE_ANN_CONV))
		return MODE_ANN_CONV;
	else if(in == trainModeToStr(MODE_ANN_SCRIPT))
		return MODE_ANN_SCRIPT;
	else if(in == trainModeToStr(MODE_AUTO_SCRIPT))
		return MODE_AUTO_SCRIPT;
	else if(in == trainModeToStr(MODE_AUTO))
		return MODE_AUTO;
	else if(in == trainModeToStr(MODE_GAN))
		return MODE_GAN;
	else if(in == trainModeToStr(MODE_REGRESSION))
		return MODE_REGRESSION;
	else if(in == trainModeToStr(MODE_REGRESSION_SCRIPT))
		return MODE_REGRESSION_SCRIPT;
	return MODE_INVALID;
}

struct Config
{
	TrainMode mode = MODE_INVALID;
	DatasetMode datasetMode = DATASET_INVALID;
	std::filesystem::path fileName;
	std::filesystem::path testFileName;
	std::filesystem::path outputDir;
	std::filesystem::path scriptPath;
	double learingRate = 0.005;
	size_t batchSize = 256;
	size_t epochs = 30;
	size_t extraLayers = 3;
	size_t latentSize = 10;
	bool noGpu = false;
	bool noWeights = false;
};

static error_t parse_opt (int key, char *arg, struct argp_state *state)
{
	Config *config = reinterpret_cast<Config*>(state->input);

	try
	{
		switch (key)
		{
		case 'q':
			Log::level = Log::ERROR;
			break;
		case 'v':
			Log::level = Log::DEBUG;
			break;
		case 'm':
			config->mode = parseTrainMode(arg);
			if(config->mode == MODE_INVALID)
			{
				Log(Log::ERROR)<<"mode has to be one of: " MODE_LIST;
				argp_usage(state);
			}
			break;
		case 'd':
			config->datasetMode = parseDatasetMode(arg);
			if(config->datasetMode == DATASET_INVALID)
			{
				Log(Log::ERROR)<<"dataset has to be one of: " DATASET_LIST;
				argp_usage(state);
			}
			break;
		case 'f':
			config->fileName.assign(arg);
			break;
		case 't':
			config->testFileName.assign(arg);
			break;
		case 'o':
			config->outputDir.assign(arg);
			break;
		case 'b':
			config->batchSize = std::stoul(std::string(arg));
			break;
		case 'i':
			config->epochs = std::stoul(std::string(arg));
			break;
		case 'l':
			config->extraLayers = std::stoul(std::string(arg));
			break;
		case 'r':
			config->learingRate = std::stod(std::string(arg));
			break;
		case 'c':
			config->noGpu = true;
			break;
		case 'n':
			config->scriptPath = arg;
			break;
		case 'g':
			config->noWeights = true;
			break;
		case 'a':
			config->latentSize = std::stoul(std::string(arg));
			break;
		default:
			return ARGP_ERR_UNKNOWN;
		}
	}
	catch(const std::invalid_argument& ex)
	{
		std::cout<<arg<<" passed for argument -"<<static_cast<char>(key)<<" is not a valid.\n";
		return ARGP_KEY_ERROR;
	}
	return 0;
}

static struct argp argp = {options, parse_opt, args_doc, doc};
