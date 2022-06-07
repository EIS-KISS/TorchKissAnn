#pragma once
#include <string>
#include <vector>
#include <argp.h>
#include <iostream>
#include "utils/log.h"

const char *argp_program_version = "TorchKissAnnTrain";
const char *argp_program_bug_address = "<carl@uvos.xyz>";
static char doc[] = "Application that trains models for TorchKissAnn";
static char args_doc[] = "";

static struct argp_option options[] =
{
  {"verbose",		'v', 0,				0,	"Show debug messages" },
  {"quiet", 		'q', 0,				0,	"only output data" },
  {"model", 		'm', "[STRING]",	0,	"model to train: ann, svm, svmcross or svmfilter"},
  {"export-dir",	'e', "[DIRECTORY]",	0,	"directory to save diagonistic data too"},
  {"export-prob",	'o', "[DIRECTORY]",	0,	"directory to save diagonistic data too"},
  {"noise",			'x', "[AMPLITUDE]",	OPTION_ARG_OPTIONAL,	"directory to save diagonistic data too"},
  { 0 }
};

typedef enum
{
	MODE_INVALID = -1,
	MODE_ANN = 0,
	MODE_SVM,
	MODE_SVM_FILTER,
	MODE_SVM_CROSS
} TrainMode;

static TrainMode parseTrainMode(const std::string& in)
{
	if(in.empty() || in == "ann")
		return MODE_ANN;
	else if(in == "svm")
		return MODE_SVM;
	else if(in == "svmcross")
		return MODE_SVM_CROSS;
	else if(in == "svmfilter")
		return MODE_SVM_FILTER;

	return MODE_INVALID;
}

struct Config
{
	TrainMode mode = MODE_INVALID;
	std::string exportDirectory;
	double noise = 0;
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
			Log(Log::ERROR)<<"mode has to be one of: ann, svm, svmcross, svmfilter";
			argp_usage(state);
		}
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
	default:
		return ARGP_ERR_UNKNOWN;
	}
	return 0;
}

static struct argp argp = {options, parse_opt, args_doc, doc};
