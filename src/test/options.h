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
#include <cstdint>
#include <string>
#include <vector>
#include <argp.h>
#include <iostream>
#include <filesystem>
#include "utils/log.h"
#include "commonoptions.h"

#define MODE_LIST "ann, conv, script, gan, regression, regression_script, autoencoder"

inline const char *argp_program_version = "TorchKissAnnTest";
inline const char *argp_program_bug_address = "<carl@uvos.xyz>";
static char doc[] = "Application that tests models for TorchKissAnn";
static char args_doc[] = "";

enum TestArgpKey
{
	OPT_FIT_BATCH = 0x81,
	OPT_FIT_RESTARTS,
	OPT_FIT_EVALUATIONS,
	OPT_FIT_LIMIT,
	OPT_FIT_POINTS,
	OPT_SIMPLICITY_THRESHOLD,
	OPT_SIMPLICITY_PREFERENCE,
};

static struct argp_option options[] =
{
  {"verbose",		'v', 0,				0,	"Show debug messages" },
  {"quiet", 		'q', 0,				0,	"only output data" },
  {"input-importance", 'p', 0,			0,	"compute imput importance"},
  {"dataset", 		'd', "[STRING]",	0,	"dataset type to use for testing: " DATASET_LIST},
  {"images", 		'g', 0,				0,	"output more graphs: regression additionally writes one Nyquist plot per spectra into <output-dir>/spectra showing the measured points against the spectra the predicted parameters simulate"},
  {"file", 			'f', "[STRING]",	0,	"filename for dataset"},
  {"batch-size",	'b', "[NUMBER]",	0,	"size of the testing batch"},
  {"output-dir",	'o', "[DIRECTORY]", 0,	"directory where testing logs will be saved"},
  {"cpu",			'c', 0,				0,	"don't use gpu even if one is available"},
  {"loss",			's', "[STRING]",	0,	"loss function to score with: spectra, permutation and composite for regression, simplicity for classification: " LOSS_LIST},
  {"no-confustion",	-1, 0,				0,	"dont compute or output the confustion matrix"},
  {"network",		'n', "[PATH]",		0,	"path to the network to test"},
  {"ignore-missmatch",	'i', 0,			0,	"Ignore missmatches in label names"},
  {"fit-mismatch",	'a', 0,				0,	"classification only: fit the circuit the network predicted to every spectra it labelled wrongly and report an after fit accuracy, the share of the test set for which the network names a circuit that actually describes the spectra. Requires a network that stores its output labels"},
  {"fit-batch-size",	OPT_FIT_BATCH, "[NUMBER]", 0, "number of mismatches fitted simultaneously, default: 64"},
  {"fit-restarts",	OPT_FIT_RESTARTS, "[NUMBER]", 0, "random restarts per mismatch on top of the start the circuit implies, default: 4. Every restart multiplies the time the after fit pass takes"},
  {"fit-evaluations",	OPT_FIT_EVALUATIONS, "[NUMBER]", 0, "maximum number of spectrum simulations per restart and batch, default: 400"},
  {"fit-limit",		OPT_FIT_LIMIT, "[FRACTION]", 0, "rms deviation below which the predicted circuit counts as plausible, default: 0.01, raised to three times the noise of the spectra when that is larger"},
  {"fit-points",	OPT_FIT_POINTS, "[NUMBER]", 0, "thin each mismatch to this many evenly spaced points for the search, default: 25, 0 fits every point. The jacobian costs one backward per point so this is the cheapest way to buy restarts: the verdict (rms against the noise floor) is still scored on the whole spectra, only the minima the optimiser explores change"},
  {"simplicity-threshold",	OPT_SIMPLICITY_THRESHOLD, "[FRACTION]", 0,	"With -s simplicity: relative change of the impedance below which removing an element of the generating circuit counts as invisible. Must be the value the network was trained with, it is not stored in the checkpoint. Default 0.01"},
  {"simplicity-preference",	OPT_SIMPLICITY_PREFERENCE, "[NUMBER]", 0,	"With -s simplicity: extra weight on the simplest accepted class when rebuilding the labels, default 1. Only changes the reported loss, the acceptable set is the same for any value"},
  {"no-progress",	OPT_NO_PROGRESS, 0,			0,	"don't draw progress bars"},
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
	bool outputMoreImages = false;
	bool confustionMatrix = true;
	bool fitMismatch = false;
	int64_t fitBatchSize = 64;
	int64_t fitRestarts = 4;
	int64_t fitEvaluations = 400;
	double fitLimit = 0.01;
	int64_t fitPoints = 25;
	LossType loss = LOSS_MSE;
	// -s simplicity: has to be the threshold the network was trained with, the checkpoint stores no
	// label semantics, only the class names
	double simplicityThreshold = 0.01;
	double simplicityPreference = 1.0;
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
		case 'g':
			config->outputMoreImages = true;
			break;
		case 'a':
			config->fitMismatch = true;
			break;
		case 's':
			config->loss = parseLossType(arg);
			if(config->loss == LOSS_INVALID)
			{
				Log(Log::ERROR)<<"loss has to be one of: " LOSS_LIST;
				argp_usage(state);
			}
			break;
			break;
		case OPT_FIT_BATCH:
			config->fitBatchSize = std::stoll(arg);
			break;
		case OPT_FIT_RESTARTS:
			config->fitRestarts = std::stoll(arg);
			break;
		case OPT_FIT_EVALUATIONS:
			config->fitEvaluations = std::stoll(arg);
			break;
		case OPT_FIT_LIMIT:
			config->fitLimit = std::stod(arg);
			break;
		case OPT_FIT_POINTS:
			config->fitPoints = std::stoll(arg);
			break;
		case OPT_SIMPLICITY_THRESHOLD:
			config->simplicityThreshold = std::stod(arg);
			break;
		case OPT_SIMPLICITY_PREFERENCE:
			config->simplicityPreference = std::stod(arg);
			break;
		case OPT_NO_PROGRESS:
			config->noProgress = true;
			break;
		case -1:
			config->confustionMatrix = false;
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
