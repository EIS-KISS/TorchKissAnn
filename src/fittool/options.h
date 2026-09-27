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
#include <argp.h>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>

#include "utils/log.h"

inline const char *argp_program_version = "TorchKissAnnFit";
inline const char *argp_program_bug_address = "<carl@uvos.xyz>";
static char doc[] = "Application that fits an equivalent circuit to a measured spectrum.\n"
	"The fit runs on the cpu, the circuit is simulated too small a problem for a gpu to help.";
static char args_doc[] = "";

enum FitArgpKey
{
	OPT_RESTARTS = 0x81,
	OPT_EVALUATIONS,
	OPT_SEED,
	OPT_FIT_LIMIT,
};

static struct argp_option options[] =
{
  {"verbose",		'v', 0,				0,	"Show debug messages" },
  {"quiet", 		'q', 0,				0,	"only output data" },
  {"file", 			'f', "[FILE]",		0,	"the spectrum to fit, an eis csv file"},
  {"model", 		'm', "[MODEL]",		0,	"the circuit to fit, e.g. r-rp or lr-p(r-w). A parameter written as a single value (r{100}) is the starting point of the fit, one written as a range (r{1~1000}) bounds it, one that is not given starts in the middle of the default range of that element"},
  {"output",		'o', "[FILE]",		0,	"write the spectrum of the fitted circuit, with the fitted model in its header, to this file"},
  {"restarts",		OPT_RESTARTS, "[NUMBER]", 0, "random restarts of the optimiser in addition to the start taken from the model string, default: 16"},
  {"evaluations",	OPT_EVALUATIONS, "[NUMBER]", 0, "maximum number of spectrum simulations per start, default: 2000"},
  {"seed",			OPT_SEED, "[NUMBER]", 0, "seed of the random restarts, a fit is reproducible for a given seed, default: 24293"},
  {"fit-limit",		OPT_FIT_LIMIT, "[FRACTION]", 0, "rms deviation from the spectra, as a fraction of the impedance, below which the model is reported to fit, default: 0.01. The limit is raised to three times the noise of the measurement when that is larger"},
  { 0 }
};

struct Config
{
	std::filesystem::path fileName;
	std::string modelString;
	std::filesystem::path outputFilename;
	int restarts = 16;
	int64_t evaluations = 2000;
	uint64_t seed = 0x5eed;
	double fitLimit = 0.01;
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
		case 'f':
			config->fileName.assign(arg);
			break;
		case 'm':
			config->modelString = arg;
			break;
		case 'o':
			config->outputFilename.assign(arg);
			break;
		case OPT_RESTARTS:
			config->restarts = std::stoi(arg);
			break;
		case OPT_EVALUATIONS:
			config->evaluations = std::stoll(arg);
			break;
		case OPT_SEED:
			config->seed = std::stoull(arg);
			break;
		case OPT_FIT_LIMIT:
			config->fitLimit = std::stod(arg);
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
