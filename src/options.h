#pragma once
#include <string>
#include <vector>
#include <argp.h>
#include <iostream>
#include "constants.h"
#include "utils/log.h"

const char *argp_program_version = "TorchKissAnn";
const char *argp_program_bug_address = "<carl@uvos.xyz>";
static char doc[] = "Application takes in EIS spectra and assigns equivalent circuts to them";
static char args_doc[] = "";

static struct argp_option options[] =
{
  {"verbose",		'v', 0,				0,	"Show debug messages" },
  {"quiet", 		'q', 0,				0,	"only output data" },
  {"network",		'n', "[FILE]",		0,	"Network file name" },
  {"input",			'i', "[FILE]",		0,	"Input file name" },
  {"type",			't', "[FORMAT]",	0,	"String identifying the file type of the input file. valid options are: csv, mpt, trash, gan" },
  {"show",			's', 0,				0,	"Show filtered input"},
  {"method",		'm', "[METHOD]",	0,	"select classifier method. Valid options are: ann, svm, knn, anntest"},
  {"pre",			'p', "[METHOD]",	0,	"choose input filter method. Valid options are: none, svm, gan"},
  {"pre-network",	'f', "[FILE]",		0,	"choose input filter network file."},
  {"export-dir",	'e', "[DIRECTORY]",	0,	"directory to save diagonistic data too"},
  {"export-prob",	'o', "[NUMMBER]",	0,	"probability that a data will be logged."},
  {"re-export",		'a', 0,				0,	"re-export."},
  { 0 }
};

typedef enum
{
	MODE_INVALID = -1,
	MODE_ANN = 0,
	MODE_ANN_TEST,
	MODE_NN,
	MODE_KNN,
	MODE_SVM
} PredictionMode;

typedef enum
{
	FILTER_INVALID = -1,
	FILTER_NONE = 0,
	FILTER_SVM,
	FILTER_GAN
} FilterMode;

struct Config
{
	std::string networkFileName;
	std::string spectraFileName;
	std::string filterFileName;
	std::string exportDir;
	double outputProb = 0.001;
	bool show = false;
	bool trash = false;
	bool reexport = false;
	PredictionMode mode = MODE_ANN;
	FilterMode filterMode = FILTER_NONE;
	FileType fileType = FILE_TYPE_CSV;
};

static PredictionMode parseMode(const std::string& in)
{
	if(in.empty() || in == "ann")
		return MODE_ANN;
	else if(in == "anntest")
		return MODE_ANN_TEST;
	else if(in == "svm")
		return MODE_SVM;
	else if(in == "nn")
		return MODE_NN;
	else if(in == "knn")
		return MODE_KNN;

	return MODE_INVALID;
}

static FilterMode parseFilterMode(const std::string& in)
{
	if(in.empty() || in == "none")
		return FILTER_NONE;
	else if(in == "svm")
		return FILTER_SVM;
	else if(in == "gan")
		return FILTER_GAN;

	return FILTER_INVALID;
}

static FileType parseFileType(const std::string& in)
{
	if(in.empty() || in == "csv")
		return FILE_TYPE_CSV;
	else if(in == "mpt")
		return FILE_TYPE_MPT;
	else if(in == "trash")
		return FILE_TYPE_TRASH;
	else if(in == "relaxis")
		return FILE_TYPE_RELAXIS;
	else if(in == "gan")
		return FILE_TYPE_GAN;

	return FILE_TYPE_INVALID;
}

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
	case 'n':
		config->networkFileName.assign(arg);
		break;
	case 'i':
		config->spectraFileName.assign(arg);
		break;
	case 't':
		config->fileType = parseFileType(arg);
		if(config->fileType == FILE_TYPE_INVALID)
		{
			std::cout<<arg<<" is not a vaild file type.\n";
			argp_usage(state);
		}
		break;
	case 'p':
		config->filterMode = parseFilterMode(arg);
		if(config->filterMode == FILTER_INVALID)
		{
			std::cout<<arg<<" is not a vaild filter method.\n";
			argp_usage(state);
		}
		break;
	case 's':
		config->show = true;
		break;
	case 'm':
		config->mode = parseMode(arg);
		if(config->mode == MODE_INVALID)
		{
			std::cout<<arg<<" is not a vaild classifier method.\n";
			 argp_usage(state);
		}
		break;
	case 'f':
		config->filterFileName.assign(arg);
		break;
	case 'e':
		config->exportDir.assign(arg);
		break;
	case 'o':
		config->outputProb = std::stod(std::string(arg));
		break;
	case 'a':
		config->reexport = true;
		break;
	default:
		return ARGP_ERR_UNKNOWN;
	}
	return 0;
}

static struct argp argp = {options, parse_opt, args_doc, doc};
