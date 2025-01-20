#pragma once
#include <string>
#include <vector>
#include <argp.h>
#include <iostream>
#include <filesystem>
#include "utils/log.h"
#include "commonoptions.h"

#define MODE_LIST "ann, conv, script, gan, regression, regression_script"

inline const char *argp_program_version = "TorchKissAnnTest";
inline const char *argp_program_bug_address = "<carl@uvos.xyz>";
static char doc[] = "Application that tests models for TorchKissAnn";
static char args_doc[] = "";

static struct argp_option options[] =
{
  {"verbose",		'v', 0,				0,	"Show debug messages" },
  {"quiet", 		'q', 0,				0,	"only output data" },
  {"input-importance", 'p', 0,			0,	"compute imput importance"},
  {"dataset", 		'd', "[STRING]",	0,	"dataset type to use for testing: " DATASET_LIST},
  {"file", 			'f', "[STRING]",	0,	"filename for dataset"},
  {"batch-size",	'b', "[NUMBER]",	0,	"size of the testing batch"},
  {"output-dir",	'o', "[DIRECTORY]", 0,	"directory where testing logs will be saved"},
  {"cpu",			'c', 0,				0,	"don't use gpu even if one is available"},
  {"network",		'n', "[PATH]",		0,	"path to the network to test"},
  {"ignore-missmatch",	'i', 0,			0,	"Ignore missmatches in label names"},
  { 0 }
};

struct Config
{
	DatasetMode datasetMode = DATASET_INVALID;
	std::filesystem::path fileName;
	std::filesystem::path outputDir = "./out";
	std::filesystem::path netpath;
	size_t batchSize = 256;
	bool noGpu = false;
	bool ignoreMissmatch = false;
	bool inputImportance = false;
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
		case 'o':
			config->outputDir.assign(arg);
			break;
		case 'b':
			config->batchSize = std::stoul(std::string(arg));
			break;
		case 'c':
			config->noGpu = true;
			break;
		case 'n':
			config->netpath = arg;
			break;
		case 'i':
			config->ignoreMissmatch = true;
			break;
		case 'p':
			config->inputImportance = true;
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
