#pragma once
#include <string>
#include <vector>
#include <argp.h>
#include "log.h"

const char *argp_program_version = "TorchKissAnn";
const char *argp_program_bug_address = "<carl@uvos.xyz>";
static char doc[] = "Application takes in EIS spectra and assigns equivalent circuts to them";
static char args_doc[] = "";

static struct argp_option options[] =
{
  {"verbose",	'v',	0,					0,	"Show debug messages" },
  {"quiet", 		'q',	0,					0,	"only output data" },
  {"network",	'n', "[STRING]",	0,	"Network file name" },
  {"input",		'i', "[STRING]",		0,	"Input file name" },
  {"type",		't', "[STRING]",		0,	"String identifying the file type of the file type valid options are: csv, mpt" },
  {"show",	's', 0,						0,	"Show filtered input"},
  { 0 }
};

struct Config
{
	std::string networkFileName;
	std::string spectraFileName;
	std::string fileType = "csv";
	bool show = false;
};

static error_t
parse_opt (int key, char *arg, struct argp_state *state)
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
		config->fileType.assign(arg);
		break;
	case 's':
		config->show = true;
		break;
	default:
		return ARGP_ERR_UNKNOWN;
	}
	return 0;
}

static struct argp argp = {options, parse_opt, args_doc, doc};
