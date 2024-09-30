#pragma once

#define DATASET_LIST "gen, gennoise, gennoise-dir, fdedup, passfail, dir, tar, genregression, tarregression, coincellhellpost, coincellhellpre"

typedef enum
{
	DATASET_INVALID = -1,
	DATASET_GEN = 0,
	DATASET_GEN_NOISE,
	DATASET_GEN_NOISE_DIR,
	DATASET_FDUP,
	DATASET_PASSFAIL,
	DATASET_DIR,
	DATASET_TAR,
	DATASET_GEN_REGRESSION,
	DATASET_TAR_REGRESSION,
	DATASET_COINCELLHELL_POSTDICT,
	DATASET_COINCELLHELL_PREDICT
} DatasetMode;

static inline constexpr const char* datasetModeToStr(const DatasetMode mode)
{
	switch(mode)
	{
		case DATASET_GEN:
			return "gen";
		case DATASET_GEN_NOISE:
			return "gennoise";
		case DATASET_GEN_NOISE_DIR:
			return "gennoise-dir";
		case DATASET_FDUP:
			return "fdedup";
		case DATASET_PASSFAIL:
			return "passfail";
		case DATASET_DIR:
			return "dir";
		case DATASET_TAR:
			return "tar";
		case DATASET_GEN_REGRESSION:
			return "genregression";
		case DATASET_TAR_REGRESSION:
			return "tarregression";
		case DATASET_COINCELLHELL_POSTDICT:
			return "coincellhellpost";
		case DATASET_COINCELLHELL_PREDICT:
			return "coincellhellpre";
		default:
			return "invalid";
	}
}

static inline DatasetMode parseDatasetMode(const std::string& in)
{
	if(in.empty() || in == datasetModeToStr(DATASET_GEN))
		return DATASET_GEN;
	else if(in == datasetModeToStr(DATASET_GEN_NOISE))
		return DATASET_GEN_NOISE;
	else if(in == datasetModeToStr(DATASET_GEN_NOISE_DIR))
			return DATASET_GEN_NOISE_DIR;
	else if(in == datasetModeToStr(DATASET_FDUP))
		return DATASET_FDUP;
	else if(in == datasetModeToStr(DATASET_PASSFAIL))
		return DATASET_PASSFAIL;
	else if(in == datasetModeToStr(DATASET_DIR))
		return DATASET_DIR;
	else if(in == datasetModeToStr(DATASET_TAR))
		return DATASET_TAR;
	else if(in == datasetModeToStr(DATASET_GEN_REGRESSION))
		return DATASET_GEN_REGRESSION;
	else if(in == datasetModeToStr(DATASET_TAR_REGRESSION))
		return DATASET_TAR_REGRESSION;
	else if(in == datasetModeToStr(DATASET_COINCELLHELL_POSTDICT))
		return DATASET_COINCELLHELL_POSTDICT;
	else if(in == datasetModeToStr(DATASET_COINCELLHELL_PREDICT))
		return DATASET_COINCELLHELL_PREDICT;
	return DATASET_INVALID;
}
