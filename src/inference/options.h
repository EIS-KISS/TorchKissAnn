/* * TorchKissAnn - A collection of tools to train various types of Machine learning
 * algorithms on various types of EIS data
 * Copyright (C) 2025 Carl Klemm <carl@uvos.xyz>
 *
 * This file is part of TorchKissAnn.
 *
 * TorchKissAnn is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * TorchKissAnn is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with TorchKissAnn.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once
#include <string>
#include <argp.h>
#include <iostream>
#include "globals.h"
#include "utils/log.h"
#include "utils/commonoptions.h"

const inline char *argp_program_version = "TorchKissAnn";
const inline char *argp_program_bug_address = "<carl@uvos.xyz>";
static char doc[] = "Application takes in EIS spectra and assigns equivalent circuts to them";
static char args_doc[] = "";

static struct argp_option options[] =
{
  {"verbose",		'v', 0,				0,	"Show debug messages" },
  {"quiet", 		'q', 0,				0,	"only output data" },
  {"network",		'n', "[FILE]",		0,	"Network file name" },
  {"input",			'i', "[FILE]",		0,	"Input file name" },
  {"dataset",	 	'd', "[STRING]",	0,	"The dataset type to test on :" DATASET_LIST},
  {"type",			't', "[FORMAT]",	0,	"String identifying the file type of the input file. valid options are: csv, trash, gen" },
  {"mode",			'm', "[MODE]",		0,	"select a mode. Valid options are: ann, knn, anntest, annconfusion, regression, show"},
  {"pre",			'p', "[METHOD]",	0,	"choose input filter method. Valid options are: none, gan"},
  {"pre-network",	'f', "[FILE]",		0,	"choose input filter network file."},
  { 0 }
};

typedef enum
{
	MODE_INVALID = -1,
	MODE_ANN = 0,
	MODE_REGRESSION,
	MODE_SHOW,
	MODE_REEXPORT
} PredictionMode;

typedef enum
{
	FILTER_INVALID = -1,
	FILTER_NONE = 0,
	FILTER_GAN
} FilterMode;

struct Config
{
	std::string networkFileName;
	std::string spectraFileName;
	std::string filterFileName;
	DatasetMode datasetMode = DATASET_INVALID;
	PredictionMode mode = MODE_ANN;
	FilterMode filterMode = FILTER_NONE;
	FileType fileType = FILE_TYPE_CSV;

};

static PredictionMode parseMode(const std::string& in)
{
	if(in.empty() || in == "ann")
		return MODE_ANN;
	else if (in == "regression")
		return MODE_REGRESSION;
	else if (in == "show")
		return MODE_SHOW;
	else if (in == "reexport")
		return MODE_REEXPORT;

	return MODE_INVALID;
}

static FilterMode parseFilterMode(const std::string& in)
{
	if(in.empty() || in == "none")
		return FILTER_NONE;
	else if(in == "gan")
		return FILTER_GAN;

	return FILTER_INVALID;
}

static FileType parseFileType(const std::string& in)
{
	if(in.empty() || in == "csv")
		return FILE_TYPE_CSV;
	else if(in == "trash")
		return FILE_TYPE_TRASH;
	else if(in == "relaxis")
		return FILE_TYPE_RELAXIS;
	else if(in == "gan")
		return FILE_TYPE_GAN;
	else if(in == "gen")
		return FILE_TYPE_GENERATE;

	return FILE_TYPE_INVALID;
}

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
		case 'n':
			config->networkFileName.assign(arg);
			break;
		case 'i':
			config->spectraFileName.assign(arg);
			break;
		case 'd':
			config->datasetMode = parseDatasetMode(arg);
			if(config->datasetMode == DATASET_INVALID)
			{
				std::cout<<arg<<" is not a vaild dataset type.\n";
				argp_usage(state);
			}
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
		default:
			return ARGP_ERR_UNKNOWN;
		}
	}
	catch(const std::invalid_argument& ex)
	{
		std::cout<<arg<<" passed for argument -"<<key<<" is not a valid number.";
		return ARGP_KEY_ERROR;
	}
	return 0;
}

static struct argp argp = {options, parse_opt, args_doc, doc};
