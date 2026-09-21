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
#include <kisstype/spectra.h>

#include "data/canonicalgrid.h"

class EisSpectraDataset
{
	inline static const std::string extraInputStr = "exip_";

	CanonicalGrid canonical;

public:
	/**
	 * Copy the active canonical grid into out and return true, or return false when canonical mode
	 * is off. Used by the training loops to record the grid in the network metadata.
	 */
	bool canonicalGridInfo(CanonicalGrid& out) const
	{
		if(!canonical.enabled)
			return false;
		out = canonical;
		return true;
	}

	/**
	 * Enable canonical grid resampling. Resolves the point count from the first spectra when the
	 * grid requests zero points and precomputes the grid frequencies. Must be called from the main
	 * thread before any worker starts calling getImpl().
	 */
	void setCanonicalGrid(const CanonicalGrid& grid);

	/**
	 * Turn frequency shift augmentation on or off. Only the training dataset enables this; the test
	 * dataset and any normalization pass leave it off so they see the unshifted grid.
	 */
	void setCanonicalTraining(bool training);

protected:
	static std::pair<std::vector<std::string>, std::vector<std::string>> getExtraInputsAndLabelNames(const eis::Spectra& spectra);
	virtual eis::Spectra loadSpectraAtIndex(size_t index) = 0;
	virtual eis::Spectra loadSpectraHeaderAtIndex(size_t index) = 0;
	virtual std::vector<std::pair<std::string, int64_t>> extraInputs();
	torch::Tensor frequencies();

	const CanonicalGrid& canonicalGrid() const;
	bool canonicalEnabled() const;

	/**
	 * Resample the given points onto the canonical grid. Returns the input unchanged when canonical
	 * mode is off. When shift augmentation is active (canonical mode, training, shift enabled) the
	 * grid is shifted by a random number of decades first, which moves the whole spectra along the
	 * frequency axis while keeping its class.
	 */
	std::vector<eis::DataPoint> resampleCanonical(const std::vector<eis::DataPoint>& data);

private:
	bool canonicalTraining = false;
};
