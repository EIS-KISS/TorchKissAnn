#pragma once
#include <string>
#include <vector>
#include <argp.h>
#include "utils/log.h"

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
  {"type",		't', "[STRING]",		0,	"String identifying the file type of the file type valid options are: csv, mpt, trash" },
  {"show",	's', 0,						0,	"Show filtered input"},
  {"nn",			'b', 0,						0,	"Use Nearest Neighbor prediction method"},
  {"knn",		'k', 0,						0,	"Use k-Nearest Nieghbor prediction method"},
  {"svm",		'm', 0,						0,	"Use support vector machine prediction method"},
  {"pre-svm",	'p', "[STRING]",			0,	"Use support vector machine as pre-ann input filter"},
  { 0 }
};

typedef enum
{
	MODE_ANN = 0,
	MODE_NN,
	MODE_KNN,
	MODE_SVM
} PredictionMode;

struct Config
{
	std::string networkFileName;
	std::string spectraFileName;
	std::string fileType = "csv";
	std::string svmPre;
	bool show = false;
	bool trash = false;
	PredictionMode mode = MODE_ANN;
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
	case 'p':
		config->svmPre.assign(arg);
		break;
	case 's':
		config->show = true;
		break;
	case 'b':
		config->mode = MODE_NN;
		break;
	case 'k':
		config->mode = MODE_KNN;
		break;
	case 'm':
		config->mode = MODE_SVM;
		break;
	default:
		return ARGP_ERR_UNKNOWN;
	}
	return 0;
}

static struct argp argp = {options, parse_opt, args_doc, doc};
