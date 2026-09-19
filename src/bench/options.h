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

inline const char *argp_program_version = "TorchKissAnnBench";
inline const char *argp_program_bug_address = "<carl@uvos.xyz>";
static char doc[] = "Benchmark the forward and backward pass throughput of a network on random data, "
	"to measure the compute cost of an architecture independently of its parameter count.";
static char args_doc[] = "";

enum BenchNetKind
{
	BENCH_NET_SIMPLE,
	BENCH_NET_CONV,
	BENCH_NET_SCRIPT
};

static inline const char* benchNetName(BenchNetKind kind)
{
	switch(kind)
	{
		case BENCH_NET_CONV:
			return "conv";
		case BENCH_NET_SCRIPT:
			return "script";
		case BENCH_NET_SIMPLE:
		default:
			return "simple";
	}
}

static struct argp_option options[] =
{
  {"verbose",		'v', 0,				0,	"Show debug messages" },
  {"quiet", 		'q', 0,				0,	"only output data" },
  {"cpu",			'c', 0,				0,	"don't use gpu even if one is available"},
  {"network",		'n', "[STRING]",		0,	"network to benchmark: simple, conv or script"},
  {"script",		's', "[PATH]",		0,	"path to a TorchScript model (with -n script)"},
  {"input-size",	'i', "[INT]",		0,	"input vector length (default 200)"},
  {"output-size",	'O', "[INT]",		0,	"output vector length (default 6)"},
  {"downsample-steps",	'D', "[INT]",	0,	"taper stages for simple/conv (default 4)"},
  {"extra-steps",	'E', "[INT]",		0,	"extra constant width stages for simple/conv (default 3)"},
  {"batch-size",	'b', "[INT]",		0,	"samples per iteration (default 256)"},
  {"warmup",		'w', "[INT]",		0,	"untimed warmup iterations (default 10)"},
  {"iterations",	'x', "[INT]",		0,	"timed iterations (default 50)"},
  {"dataset-size",	'S', "[INT]",		0,	"if set, also estimate the epoch time for this many training samples"},
  {"csv",			'o', "[PATH]",		0,	"append the result as one CSV row to this file"},
  { 0 }
};

struct Config
{
	BenchNetKind netKind = BENCH_NET_SIMPLE;
	std::filesystem::path scriptPath;
	int64_t inputSize = 200;
	int64_t outputSize = 6;
	size_t downsampleSteps = 4;
	size_t extraSteps = 3;
	int batch = 256;
	int warmup = 256;
	int iterations = 512;
	int64_t datasetSize = 0;
	std::filesystem::path csvPath;
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
		case 'c':
			config->noGpu = true;
			break;
		case 'n':
		{
			std::string kind = arg;
			if(kind == "simple")
				config->netKind = BENCH_NET_SIMPLE;
			else if(kind == "conv")
				config->netKind = BENCH_NET_CONV;
			else if(kind == "script")
				config->netKind = BENCH_NET_SCRIPT;
			else
			{
				Log(Log::ERROR)<<"network has to be one of: simple, conv, script";
				argp_usage(state);
			}
			break;
		}
		case 's':
			config->scriptPath = arg;
			break;
		case 'i':
			config->inputSize = std::stoll(arg);
			break;
		case 'O':
			config->outputSize = std::stoll(arg);
			break;
		case 'D':
			config->downsampleSteps = std::stoul(arg);
			break;
		case 'E':
			config->extraSteps = std::stoul(arg);
			break;
		case 'b':
			config->batch = std::stoi(arg);
			break;
		case 'w':
			config->warmup = std::stoi(arg);
			break;
		case 'x':
			config->iterations = std::stoi(arg);
			break;
		case 'S':
			config->datasetSize = std::stoll(arg);
			break;
		case 'o':
			config->csvPath = arg;
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
