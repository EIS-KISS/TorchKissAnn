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
#include <filesystem>
#include "utils/log.h"
#include "commonoptions.h"

inline const char *argp_program_version = "TorchKissAnnGenLatent";
inline const char *argp_program_bug_address = "<carl@uvos.xyz>";
static char doc[] = "Application that genertes latent maps LatentEis";
static char args_doc[] = "";

static struct argp_option options[] =
{
  {"verbose",		'v', 0,				0,	"Show debug messages" },
  {"quiet", 		'q', 0,				0,	"only output data" },
  {"dataset", 		'd', "[STRING]",	0,	"dataset type to use: dir or tar"},
  {"file", 			'f', "[STRING]",	0,	"filename for dataset"},
  {"output",		'o', "[PATH]",		0,	"Path where to save the latent csv"},
  {"cpu",			'c', 0,				0,	"don't use gpu even if one is available"},
  {"network",		'n', "[PATH]",		0,	"path to the network to test"},
  {"no-progress",	OPT_NO_PROGRESS, 0,			0,	"don't draw progress bars"},
  { 0 }
};

struct Config
{
	DatasetMode datasetMode = DATASET_INVALID;
	std::filesystem::path fileName;
	std::filesystem::path outputFilename = "./latent.csv";
	std::filesystem::path netpath;
	bool noGpu = false;
	bool noProgress = false;
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
			config->outputFilename.assign(arg);
			break;
		case 'c':
			config->noGpu = true;
			break;
		case 'n':
			config->netpath = arg;
			break;
		case OPT_NO_PROGRESS:
			config->noProgress = true;
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
