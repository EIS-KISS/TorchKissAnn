#pragma once
#include <string>
#include <argp.h>
#include <iostream>
#include <filesystem>
#include "utils/log.h"
#include "commonoptions.h"

const inline char *argp_program_version = "TorchKissAnnTrain";
const inline char *argp_program_bug_address = "<carl@uvos.xyz>";
static char doc[] = "Application that trains models for TorchKissAnn";
static char args_doc[] = "";

static struct argp_option options[] =
{
  {"verbose",		'v', 0,				0,	"Show debug messages" },
  {"quiet", 		'q', 0,				0,	"only output data" },
  {"dataset", 		'd', "[STRING]",	0,	"dataset to report on: " DATASET_LIST},
  {"dataset-size",	's', "[NUMBER]",	0,	"the size of the training dataset to use when using a generated or semi-generated dataset"},
  {"file", 			'f', "[STRING]",	0,	"filename for dataset"},
  {"report",		'r', "[FILENAME]",	0,	"filename for the report"},
  {"images",		'i', "[DIRECTORY]",	0,	"directory to save class images into"},
  { 0 }
};

struct Config
{
	DatasetMode datasetMode = DATASET_INVALID;
	std::filesystem::path reportFileName;
	std::filesystem::path fileName;
	std::filesystem::path testFileName;
	std::filesystem::path imageOutput;
	size_t datasetSize = 2e6;
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
		case 's':
			config->datasetSize = std::stoul(std::string(arg));
			break;
		case 'r':
			config->reportFileName.assign(arg);
			break;
		case 'i':
			config->imageOutput.assign(arg);
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
