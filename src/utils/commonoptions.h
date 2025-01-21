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
