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
#include "commonoptions.h"

#define MODE_LIST "ann, conv, script, gan, regression, regression_script, autoencoder, vae, vae_simple"

inline const char *argp_program_version = "TorchKissAnnTrain";
inline const char *argp_program_bug_address = "<carl@uvos.xyz>";
static char doc[] = "Application that trains models for TorchKissAnn";
static char args_doc[] = "";

enum
{
	OPT_CANONICAL_FMIN = 0x100,
	OPT_CANONICAL_FMAX,
	OPT_CANONICAL_POINTS,
	OPT_CANONICAL_LINEAR,
	OPT_CANONICAL_AUGMENT_SHIFT,
	OPT_CANONICAL_AUGMENT_WINDOW,
	OPT_SPECTRA_NORM,
	OPT_SIMPLICITY_THRESHOLD,
	OPT_SIMPLICITY_PREFERENCE,
	OPT_AUGMENTATION_MODEL,
	OPT_SPECTRAL_WEIGHT,
	OPT_CONTRACTIVE_WEIGHT
};

static struct argp_option options[] =
{
  {"verbose",		'v', 0,				0,	"Show debug messages" },
  {"quiet", 		'q', 0,				0,	"only output data" },
  {"model", 		'm', "[STRING]",	0,	"model to train: " MODE_LIST},
  {"dataset", 		'd', "[STRING]",	0,	"dataset type to use for training: " DATASET_LIST},
  {"file", 			'f', "[STRING]",	0,	"filename for dataset"},
  {"test",			't', "[STRING]",	0,	"filename for the test dataset"},
  {"batch-size",	'b', "[NUMBER]",	0,	"size of the training batch"},
  {"epochs",		'i', "[NUMBER]",	0,	"maximum number of epochs to train, may exit earlier due to lack of progress"},
  {"extra-layers",	'l', "[NUMBER]",	0,	"extra hidden layers to use in ann and gan, default: 3"},
  {"learing-rate",	'r', "[NUMBER]",	0,	"sgd/adam lering rate, default: 0.005"},
  {"loss",			's', "[STRING]",	0,	"loss function to use: default (nll for classification, mse for regression), nll for classification, mse, spectra, permutation and composite for regression, simplicity for classification: " LOSS_LIST},
  {"output-dir",	'o', "[DIRECTORY]", 0,	"directory where training logs and models will be saved"},
  {"cpu",			'c', 0,				0,	"don't use gpu even if one is available"},
  {"network",		'n', "[PATH]",		0,	"torchScript network to train"},
  {"no-weights",	'g', 0,				0, 	"Don't use class weights"},
  {"latent-size",	'a', "[NUMBER]",	0, 	"Size of the latent vector for the autoencoder"},
  {"kl-beta",		'k', "[NUMBER]",	0, 	"Weight of the KL divergence term in the vae loss, default: 1"},
  {"spectral-weight",	OPT_SPECTRAL_WEIGHT, "[NUMBER]", 0, "Autoencoder and vae only: mix in a spectral prediction auxiliary loss with this weight (default 0 = off). A random subset of frequency points is masked from the input, the partial spectra is encoded and decoded, and the error on the held-out points is added to the loss, forcing the latent to encode the impedance curve rather than only cluster identity"},
  {"contractive-weight",	OPT_CONTRACTIVE_WEIGHT, "[NUMBER]", 0, "Autoencoder and vae only: mix in a locality preserving auxiliary loss with this weight (default 0 = off). For every spectra the 8 nearest neighbours in input space are found and the latent distances of those pairs are minimised, so similar spectra stay close in the latent space. Unlike a cluster compactness term it has no attractor points and no cluster count, so it sharpens neighbourhood structure without absorbing novel topologies into known clusters"},
  {"canonical-fmin",		OPT_CANONICAL_FMIN, "[NUMBER]", 0, "Lower edge of the canonical grid (rad/s, same unit as the data's omega column); enables canonical resampling together with --canonical-fmax"},
  {"canonical-fmax",		OPT_CANONICAL_FMAX, "[NUMBER]", 0, "Upper frequency bound (Hz) of the canonical grid"},
  {"canonical-points",		OPT_CANONICAL_POINTS, "[NUMBER]", 0, "Number of points on the canonical grid (default: taken from the first spectra)"},
  {"canonical-linear",		OPT_CANONICAL_LINEAR, 0, 0, "Use a linearly spaced canonical grid instead of logarithmic"},
  {"canonical-augment-shift",	OPT_CANONICAL_AUGMENT_SHIFT, "[NUMBER]", 0, "Classification only: randomly shift each training spectra along the log frequency axis by up to this many decades (default when canonical is on: 1, set 0 to disable)"},
  {"canonical-augment-window",	OPT_CANONICAL_AUGMENT_WINDOW, "[NUMBER]", 0, "Crop each training spectra to a random contiguous sub-range keeping at least this fraction of the points, then resample (valid for regression too; default 1 = disabled, e.g. 0.5 crops down to half)"},
  {"spectra-norm",	OPT_SPECTRA_NORM,	"[STRING]", 0,	"Classification only: divide each spectra by a reference impedance so only the shape of the Nyquist plot reaches the network: none, zmean (mean |Z|), zgeom (geometric mean |Z|), zmax (largest |Z|). Default none. Recorded in the checkpoint so _test and libkissinference apply the same scaling"},
  {"augmentation-model",	OPT_AUGMENTATION_MODEL,	"[STRING]", 0,	"Training augmentation: add a simulated artifact to each training spectra, generated by eisgenerator from a model string that carries a parameter range. A random sweep index is drawn per spectra and that parameter combination's impedance is added to every point; 25% of the spectra are left clean. The test dataset is untouched. Use it to make a simulated model survive transfer onto measured data that carries the artifact. Example (series-inductance tail, Im += omega*L with L uniform in [9e-8, 6e-7] H): --augmentation-model \"l{9e-8~6e-7:1000}\""},
  {"simplicity-threshold",	OPT_SIMPLICITY_THRESHOLD, "[FRACTION]", 0,	"With -s simplicity: relative change of |Z| below which removing elements of the generating circuit counts as invisible, so that the simpler circuit left behind is an equally correct answer. Judged as the worst deviation anywhere on the frequency band, so a value of 0.01 excuses a reduction that never moves the spectra by more than 1%. Default 0.01, set it lower to excuse only reductions that are truly unmeasurable"},
  {"simplicity-preference",	OPT_SIMPLICITY_PREFERENCE, "[NUMBER]", 0,	"With -s simplicity: extra weight on the simplest accepted class, default 1. 0 leaves the network indifferent between the generating circuit and the simpler ones that describe the same spectra, larger values push harder towards always naming the simplest circuit that still fits"},
  {"no-progress",	OPT_NO_PROGRESS, 0,			0,	"don't draw progress bars"},
  { 0 }
};

struct Config
{
	TrainMode mode = MODE_INVALID;
	DatasetMode datasetMode = DATASET_INVALID;
	std::filesystem::path fileName;
	std::filesystem::path testFileName;
	std::filesystem::path outputDir;
	std::filesystem::path scriptPath;
	double learingRate = 0.005;
	size_t batchSize = 256;
	size_t epochs = 30;
	size_t extraLayers = 3;
	size_t latentSize = 10;
	double klBeta = 1.0;
	double spectralWeight = 0;
	double contractiveWeight = 0;
	LossType loss = LOSS_DEFAULT;
	bool noGpu = false;
	bool noWeights = false;
	// Canonical frequency grid: resample every spectra onto a fixed grid so the same
	// feature lands at the same input index regardless of the measured frequency range.
	// Enabled when both canonicalFMin and canonicalFMax are given (see check_options).
	bool canonical = false;
	double canonicalFMin = 0;
	double canonicalFMax = 0;
	size_t canonicalPoints = 0;
	bool canonicalLog = true;
	double canonicalAugmentMaxShift = 1.0;   // decades; 0 disables shift augmentation
	double canonicalAugmentWindowMin = 1.0;  // minimum fraction of points kept when windowing; 1 disables
	bool canonicalAugmentShiftRequested = false; // set when --canonical-augment-shift was passed explicitly
	// Per spectra |Z| scaling for classification: name of the reference impedance each spectra is
	// divided by (see SpectraNorm). "none" leaves the raw impedances alone.
	std::string spectraNorm = "none";
	// eisgenerator artifact augmentation (training only, see EisSpectraDataset): a model string with an
	// embedded parameter range whose random sweep index is added to each training spectra. Empty = off.
	std::string augmentationModel;
	// -s simplicity: how far an element has to drop out of the impedance for a circuit missing it to
	// be an acceptable answer, and how much the simplest accepted class is preferred over the others.
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
		case 'm':
			config->mode = parseTrainMode(arg);
			if(config->mode == MODE_INVALID)
			{
				Log(Log::ERROR)<<"mode has to be one of: " MODE_LIST;
				argp_usage(state);
			}
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
		case 't':
			config->testFileName.assign(arg);
			break;
		case 'o':
			config->outputDir.assign(arg);
			break;
		case 'b':
			config->batchSize = std::stoul(std::string(arg));
			break;
		case 'i':
			config->epochs = std::stoul(std::string(arg));
			break;
		case 'l':
			config->extraLayers = std::stoul(std::string(arg));
			break;
		case 'r':
			config->learingRate = std::stod(std::string(arg));
			break;
		case 's':
			config->loss = parseLossType(arg);
			if(config->loss == LOSS_INVALID)
			{
				Log(Log::ERROR)<<"loss has to be one of: " LOSS_LIST;
				argp_usage(state);
			}
			break;
		case 'c':
			config->noGpu = true;
			break;
		case 'n':
			config->scriptPath = arg;
			break;
		case 'g':
			config->noWeights = true;
			break;
		case 'a':
			config->latentSize = std::stoul(std::string(arg));
			break;
		case 'k':
			config->klBeta = std::stod(std::string(arg));
			break;
		case OPT_SPECTRAL_WEIGHT:
			config->spectralWeight = std::stod(std::string(arg));
			break;
		case OPT_CONTRACTIVE_WEIGHT:
			config->contractiveWeight = std::stod(std::string(arg));
			break;
		case OPT_CANONICAL_FMIN:
			config->canonicalFMin = std::stod(std::string(arg));
			break;
		case OPT_CANONICAL_FMAX:
			config->canonicalFMax = std::stod(std::string(arg));
			break;
		case OPT_CANONICAL_POINTS:
			config->canonicalPoints = std::stoul(std::string(arg));
			break;
		case OPT_CANONICAL_LINEAR:
			config->canonicalLog = false;
			break;
		case OPT_CANONICAL_AUGMENT_SHIFT:
			config->canonicalAugmentMaxShift = std::stod(std::string(arg));
			config->canonicalAugmentShiftRequested = true;
			break;
		case OPT_CANONICAL_AUGMENT_WINDOW:
			config->canonicalAugmentWindowMin = std::stod(std::string(arg));
			break;
		case OPT_SPECTRA_NORM:
			config->spectraNorm = arg;
			break;
		case OPT_AUGMENTATION_MODEL:
			config->augmentationModel = arg;
			break;
		case OPT_SIMPLICITY_THRESHOLD:
			config->simplicityThreshold = std::stod(std::string(arg));
			break;
		case OPT_SIMPLICITY_PREFERENCE:
			config->simplicityPreference = std::stod(std::string(arg));
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
		std::cout<<arg<<" passed for argument -"<<static_cast<char>(key)<<" is not a valid.\n";
		return ARGP_KEY_ERROR;
	}
	return 0;
}

static struct argp argp = {options, parse_opt, args_doc, doc};
