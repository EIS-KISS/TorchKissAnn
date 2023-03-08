#pragma once
#include <string>
#include <vector>
#include <argp.h>
#include <iostream>
#include <filesystem>
#include "utils/log.h"

#define MODE_LIST "ann, conv, gan"
#define DATASET_LIST "gen, fdedup, dir, trash"

const char *argp_program_version = "TorchKissAnnTrain";
const char *argp_program_bug_address = "<carl@uvos.xyz>";
static char doc[] = "Application that trains models for TorchKissAnn";
static char args_doc[] = "";

static struct argp_option options[] =
{
  {"verbose",		'v', 0,				0,	"Show debug messages" },
  {"quiet", 		'q', 0,				0,	"only output data" },
  {"model", 		'm', "[STRING]",	0,	"model to train: " MODE_LIST},
  {"dataset", 		'd', "[STRING]",	0,	"model to use for training: " DATASET_LIST},
  {"dataset-size",	's', "[NUMBER]",	0,	"the size of the training dataset to use when using a generated or semi-generated dataset"},
  {"file", 			'f', "[STRING]",	0,	"filename for dataset"},
  {"test",			't', "[STRING]",	0,	"filename for the test dataset"},
  {"export-dir",	'e', "[DIRECTORY]",	0,	"directory to save diagonistic data too"},
  {"export-prob",	'o', "[NUMMBER]",	0,	"probability that a data will be logged."},
  {"noise",			'x', "[AMPLITUDE]",	0,	"noise amplitude to be applied to generated spectra"},
  {"batch-size",	'b', "[NUMBER]",	0,	"size of the training batch"},
  {"epochs",		'i', "[NUMBER]",	0,	"maximum number of epochs to train, may exit earlier due to lack of progress"},
  {"probe-model",	'p', 0,				0,	"use probe model (l{5e-7})"},
  {"extra-layers",	'l', "[NUMBER]",	0,	"extra hidden layers to use in ann and gan, default: 3"},
  {"learing-rate",	'r', "[NUMBER]",	0,	"sgd lering rate, default: 0.005"},
  { 0 }
};

typedef enum
{
	MODE_INVALID = -1,
	MODE_ANN = 0,
	MODE_ANN_CONV,
	MODE_GAN
} TrainMode;

typedef enum
{
	DATASET_INVALID = -1,
	DATASET_GEN = 0,
	DATASET_FDUP,
	DATASET_DIR,
	DATASET_TRASH
} DatasetMode;

static TrainMode parseTrainMode(const std::string& in)
{
	if(in.empty() || in == "ann")
		return MODE_ANN;
	else if(in == "conv")
		return MODE_ANN_CONV;
	else if(in == "gan")
		return MODE_GAN;

	return MODE_INVALID;
}

static DatasetMode parseDatasetMode(const std::string& in)
{
	if(in.empty() || in == "gen")
		return DATASET_GEN;
	else if(in == "fdedup")
		return DATASET_FDUP;
	else if(in == "dir")
		return DATASET_DIR;
	else if(in == "trash")
		return DATASET_TRASH;

	return DATASET_INVALID;
}

struct Config
{
	TrainMode mode = MODE_INVALID;
	DatasetMode datasetMode = DATASET_INVALID;
	std::string exportDirectory;
	std::filesystem::path fileName;
	std::filesystem::path testFileName;
	double noise = 0;
	double learingRate = 0.005;
	size_t batchSize = 100;
	size_t epochs = 30;
	size_t datasetSize = 2e6;
	size_t extraLayers = 3;
	bool probeModel = false;
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
		case 'e':
			config->exportDirectory.assign(arg);
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
