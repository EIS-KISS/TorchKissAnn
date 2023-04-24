#pragma once
#include <string>
#include <vector>
#include <argp.h>
#include <iostream>
#include <filesystem>
#include "utils/log.h"

#define MODE_LIST "ann, conv, gan, regression"
#define DATASET_LIST "gen, fdedup, dir, trash, genregression"

const char *argp_program_version = "TorchKissAnnTrain";
const char *argp_program_bug_address = "<carl@uvos.xyz>";
static char doc[] = "Application that trains models for TorchKissAnn";
static char args_doc[] = "";

static struct argp_option options[] =
{
  {"verbose",		'v', 0,				0,	"Show debug messages" },
  {"quiet", 		'q', 0,				0,	"only output data" },
  {"model", 		'm', "[STRING]",	0,	"model to train: " MODE_LIST},
  {"dataset", 		'd', "[STRING]",	0,	"dataset type to use for training: " DATASET_LIST},
  {"dataset-size",	's', "[NUMBER]",	0,	"the size of the training dataset to use when using a generated or semi-generated dataset"},
  {"file", 			'f', "[STRING]",	0,	"filename for dataset"},
  {"test",			't', "[STRING]",	0,	"filename for the test dataset"},
  {"noise",			'x', "[AMPLITUDE]",	0,	"noise amplitude to be applied to generated spectra"},
  {"batch-size",	'b', "[NUMBER]",	0,	"size of the training batch"},
  {"epochs",		'i', "[NUMBER]",	0,	"maximum number of epochs to train, may exit earlier due to lack of progress"},
  {"probe-model",	'p', 0,				0,	"use probe model (l{5e-7})"},
  {"extra-layers",	'l', "[NUMBER]",	0,	"extra hidden layers to use in ann and gan, default: 3"},
  {"learing-rate",	'r', "[NUMBER]",	0,	"sgd/adam lering rate, default: 0.005"},
  {"output-dir",	'o', "[DIRECTORY]", 0,	"directory where training logs and models will be saved"},
  {"no-gpu",		'n', 0,				0,	"don't use gpu even if one is available"},
  { 0 }
};

typedef enum
{
	MODE_INVALID = -1,
	MODE_ANN = 0,
	MODE_ANN_CONV,
	MODE_GAN,
	MODE_REGRESSION
} TrainMode;

typedef enum
{
	DATASET_INVALID = -1,
	DATASET_GEN = 0,
	DATASET_FDUP,
	DATASET_DIR,
	DATASET_TRASH,
	DATASET_GEN_REGRESSION
} DatasetMode;

static std::string trainModeToStr(const TrainMode mode)
{
	switch(mode)
	{
		case MODE_ANN:
			return "ann";
		case MODE_ANN_CONV:
			return "conv";
		case MODE_GAN:
			return "gan";
		case MODE_REGRESSION:
			return "regression";
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
	else if(in == trainModeToStr(MODE_GAN))
		return MODE_GAN;
	else if(in == trainModeToStr(MODE_REGRESSION))
		return MODE_REGRESSION;
	return MODE_INVALID;
}

static std::string datasetModeToStr(const DatasetMode mode)
{
	switch(mode)
	{
		case DATASET_GEN:
			return "gen";
		case DATASET_FDUP:
			return "fdedup";
		case DATASET_DIR:
			return "dir";
		case DATASET_TRASH:
			return "trash";
		case DATASET_GEN_REGRESSION:
			return "genregression";
		default:
			return "invalid";
	}
}

static DatasetMode parseDatasetMode(const std::string& in)
{
	if(in.empty() || in == datasetModeToStr(DATASET_GEN))
		return DATASET_GEN;
	else if(in == datasetModeToStr(DATASET_FDUP))
		return DATASET_FDUP;
	else if(in == datasetModeToStr(DATASET_DIR))
		return DATASET_DIR;
	else if(in == datasetModeToStr(DATASET_TRASH))
		return DATASET_TRASH;
	else if(in == datasetModeToStr(DATASET_GEN_REGRESSION))
		return DATASET_GEN_REGRESSION;
	return DATASET_INVALID;
}

struct Config
{
	TrainMode mode = MODE_INVALID;
	DatasetMode datasetMode = DATASET_INVALID;
	std::filesystem::path fileName;
	std::filesystem::path testFileName;
	std::filesystem::path outputDir;
	double noise = 0;
	double learingRate = 0.005;
	size_t batchSize = 256;
	size_t epochs = 30;
	size_t datasetSize = 2e6;
	size_t extraLayers = 3;
	bool probeModel = false;
	bool noGpu = false;
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
		case 'x':
			if(arg)
				config->noise = std::stod(std::string(arg));
			else
				config->noise = 0.02;
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
		case 's':
			config->datasetSize = std::stoul(std::string(arg));
			break;
		case 'p':
			config->probeModel = true;
			break;
		case 'l':
			config->extraLayers = std::stoul(std::string(arg));
			break;
		case 'r':
			config->learingRate = std::stod(std::string(arg));
			break;
		case 'n':
			config->noGpu = true;
			break;
		default:
			return ARGP_ERR_UNKNOWN;
		}
	}
	catch(const std::invalid_argument& ex)
	{
		std::cout<<arg<<" passed for argument -"<<static_cast<char>(key)<<" is not a valid number.\n";
		return ARGP_KEY_ERROR;
	}
	return 0;
}

static struct argp argp = {options, parse_opt, args_doc, doc};
