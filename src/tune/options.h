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
#include <vector>
#include <argp.h>
#include <iostream>
#include <filesystem>
#include "utils/log.h"

#define MODE_LIST "ann, conv, gan, regression"
#define DATASET_LIST "gen, fdedup, dir, tar, trash, genregression, tarregression"

const char *argp_program_version = "TorchKissAnnTune";
const char *argp_program_bug_address = "<carl@uvos.xyz>";
static char doc[] = "Application that tunes model hyperparameters for TorchKissAnn";
static char args_doc[] = "";

static struct argp_option options[] =
{
  {"verbose",		'v', 0,				0,	"Show debug messages" },
  {"quiet", 		'q', 0,				0,	"only output data" },
  {"model", 		'm', "[STRING]",	0,	"model to train: " MODE_LIST},
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

struct Config
{
	TrainMode mode = MODE_INVALID;
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
