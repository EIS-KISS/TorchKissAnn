#pragma once
#include <string>
#include <vector>
#include <argp.h>
#include <iostream>
#include "utils/log.h"

#define MODE_LIST "ann, conv, gan, svm, svmfilter, svmcross"
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
  {"file", 			'f', "[STRING]",	0,	"filename for dataset"},
  {"test",			't', "[STRING]",	0,	"filename for the test dataset"},
  {"export-dir",	'e', "[DIRECTORY]",	0,	"directory to save diagonistic data too"},
  {"export-prob",	'o', "[NUMMBER]",	0,	"probability that a data will be logged."},
  {"noise",			'x', "[AMPLITUDE]",	0,	"noise amplitude to be applied to generated spectra"},
  {"batch-size",	'b', "[NUMBER]",	0,	"size of the training batch"},
  {"epochs",		'i', "[NUMBER]",	0,	"maximum number of epochs to train, may exit earlier due to lack of progress"},
  { 0 }
};

typedef enum
{
	MODE_INVALID = -1,
	MODE_ANN = 0,
	MODE_ANN_CONV,
	MODE_SVM,
	MODE_SVM_FILTER,
	MODE_SVM_CROSS,
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
	else if(in == "svm")
		return MODE_SVM;
	else if(in == "svmcross")
		return MODE_SVM_CROSS;
	else if(in == "svmfilter")
		return MODE_SVM_FILTER;
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
	std::string fileName;
	std::string testFileName;
	double noise = 0;
	size_t batchSize = 100;
	size_t epochs = 3;
};

static error_t parse_opt (int key, char *arg, struct argp_state *state)
{
	Config *config = reinterpret_cast<Config*>(state->input);

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
	default:
		return ARGP_ERR_UNKNOWN;
	}
	return 0;
}

static struct argp argp = {options, parse_opt, args_doc, doc};
