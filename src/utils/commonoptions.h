#pragma once

#define DATASET_LIST "dir, tar, dirreg, tarreg"

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
