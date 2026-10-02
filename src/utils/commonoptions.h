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

#define DATASET_LIST "dir, tar, dirreg, tarreg"
#define LOSS_LIST "default, nll, mse, spectra, permutation, composite, simplicity"

// argp key for the options shared by every tool. The per tool enums use 0x81 and 0x100 as their
// bases, so this one starts above them.
enum CommonArgpKey
{
	OPT_NO_PROGRESS = 0x200,
};

typedef enum
{
	DATASET_INVALID = -1,
	DATASET_DIR,
	DATASET_TAR,
	DATASET_DIR_REGRESSION,
	DATASET_TAR_REGRESSION
} DatasetMode;

static inline constexpr const char* datasetModeToStr(const DatasetMode mode)
{
	switch(mode)
	{
		case DATASET_DIR:
			return "dir";
		case DATASET_TAR:
			return "tar";
		case DATASET_DIR_REGRESSION:
			return "dirreg";
		case DATASET_TAR_REGRESSION:
			return "tarreg";
		default:
			return "invalid";
	}
}

// True for the two flavours that carry a continuous parameter vector as their target, false for
// the ones that carry an interned circuit class.
static inline constexpr bool datasetModeIsRegression(const DatasetMode mode)
{
	return (mode == DATASET_DIR_REGRESSION) || (mode == DATASET_TAR_REGRESSION);
}

static inline DatasetMode parseDatasetMode(const std::string& in)
{
	if(in == datasetModeToStr(DATASET_DIR))
		return DATASET_DIR;
	else if(in == datasetModeToStr(DATASET_TAR))
		return DATASET_TAR;
	else if(in == datasetModeToStr(DATASET_DIR_REGRESSION))
		return DATASET_DIR_REGRESSION;
	else if(in == datasetModeToStr(DATASET_TAR_REGRESSION))
		return DATASET_TAR_REGRESSION;
	return DATASET_INVALID;
}


typedef enum
{
	LOSS_INVALID = -1,
	LOSS_MSE = 0,
	LOSS_SPECTRA,
	LOSS_PERMUTATION,
	LOSS_COMPOSITE,
	LOSS_SIMPLICITY,
	LOSS_NLL,
	LOSS_DEFAULT
} LossType;

static inline std::string lossTypeToStr(const LossType loss)
{
	switch(loss)
	{
		case LOSS_MSE:
			return "mse";
		case LOSS_SPECTRA:
			return "spectra";
		case LOSS_PERMUTATION:
			return "permutation";
		case LOSS_COMPOSITE:
			return "composite";
		case LOSS_SIMPLICITY:
			return "simplicity";
		case LOSS_NLL:
			return "nll";
		case LOSS_DEFAULT:
			return "default";
		default:
			return "Invalid";
	}
}

static inline LossType parseLossType(const std::string& in)
{
	if(in.empty() || in == lossTypeToStr(LOSS_DEFAULT))
		return LOSS_DEFAULT;
	else if(in == lossTypeToStr(LOSS_NLL))
		return LOSS_NLL;
	else if(in == lossTypeToStr(LOSS_MSE))
		return LOSS_MSE;
	else if(in == lossTypeToStr(LOSS_SPECTRA))
		return LOSS_SPECTRA;
	else if(in == lossTypeToStr(LOSS_PERMUTATION))
		return LOSS_PERMUTATION;
	else if(in == lossTypeToStr(LOSS_COMPOSITE))
		return LOSS_COMPOSITE;
	else if(in == lossTypeToStr(LOSS_SIMPLICITY))
		return LOSS_SIMPLICITY;
	return LOSS_INVALID;
}

static inline constexpr bool lossIsRegressionOnly(const LossType loss)
{
	return loss == LOSS_MSE || loss == LOSS_SPECTRA || loss == LOSS_PERMUTATION || loss == LOSS_COMPOSITE;
}

static inline constexpr bool lossIsClassificationOnly(const LossType loss)
{
	return loss == LOSS_NLL || loss == LOSS_SIMPLICITY;
}

typedef enum
{
	MODE_INVALID = -1,
	MODE_ANN = 0,
	MODE_ANN_CONV,
	MODE_ANN_SCRIPT,
	MODE_AUTO_SCRIPT,
	MODE_AUTO,
	MODE_GAN,
	MODE_REGRESSION,
	MODE_REGRESSION_SCRIPT,
	MODE_VAE_SCRIPT,
	MODE_VAE
} TrainMode;

static inline std::string trainModeToStr(const TrainMode mode)
{
	switch(mode)
	{
		case MODE_ANN:
			return "ann";
		case MODE_ANN_CONV:
			return "conv";
		case MODE_ANN_SCRIPT:
			return "script";
		case MODE_AUTO_SCRIPT:
			return "autoencoder";
		case MODE_AUTO:
			return "autoencoder_simple";
		case MODE_GAN:
			return "gan";
		case MODE_REGRESSION:
			return "regression";
		case MODE_REGRESSION_SCRIPT:
			return "regression_script";
		case MODE_VAE_SCRIPT:
			return "vae";
		case MODE_VAE:
			return "vae_simple";
		default:
			return "Invalid";
	}
}

static inline TrainMode parseTrainMode(const std::string& in)
{
	if(in.empty() || in == trainModeToStr(MODE_ANN))
		return MODE_ANN;
	else if(in == trainModeToStr(MODE_ANN_CONV))
		return MODE_ANN_CONV;
	else if(in == trainModeToStr(MODE_ANN_SCRIPT))
		return MODE_ANN_SCRIPT;
	else if(in == trainModeToStr(MODE_AUTO_SCRIPT))
		return MODE_AUTO_SCRIPT;
	else if(in == trainModeToStr(MODE_AUTO))
		return MODE_AUTO;
	else if(in == trainModeToStr(MODE_GAN))
		return MODE_GAN;
	else if(in == trainModeToStr(MODE_REGRESSION))
		return MODE_REGRESSION;
	else if(in == trainModeToStr(MODE_REGRESSION_SCRIPT))
		return MODE_REGRESSION_SCRIPT;
	else if(in == trainModeToStr(MODE_VAE_SCRIPT))
		return MODE_VAE_SCRIPT;
	else if(in == trainModeToStr(MODE_VAE))
		return MODE_VAE;
	return MODE_INVALID;
}

static inline bool modeIsClassification(const TrainMode mode)
{
	return mode == MODE_ANN || mode == MODE_ANN_CONV || mode == MODE_ANN_SCRIPT;
}

static inline bool modeIsRegression(const TrainMode mode)
{
	return mode == MODE_REGRESSION || mode == MODE_REGRESSION_SCRIPT;
}

static inline bool modeIsAutoEncoder(const TrainMode mode)
{
	return mode == MODE_AUTO_SCRIPT || mode == MODE_AUTO || mode == MODE_VAE_SCRIPT || mode == MODE_VAE;
}

static inline bool modeIsVariational(const TrainMode mode)
{
	return mode == MODE_VAE_SCRIPT || mode == MODE_VAE;
}

static inline bool modeIsScript(const TrainMode mode)
{
	return mode == MODE_ANN_SCRIPT || mode == MODE_REGRESSION_SCRIPT || mode == MODE_AUTO_SCRIPT || mode == MODE_VAE_SCRIPT;
}
