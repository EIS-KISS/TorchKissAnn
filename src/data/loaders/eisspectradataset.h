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
#include <kisstype/spectra.h>

#include "data/canonicalgrid.h"

/**
 * How a single spectra is scaled to a common impedance level before it reaches the network.
 *
 * Every point of the spectra is divided by one reference impedance, so the shape of the Nyquist plot
 * is untouched and only its absolute |Z| level is removed. That is what makes the transform usable
 * for classification: the circuit topology, and therefore the class, is preserved, while the
 * difference between, say, a 1 mm and a 10 mm electrode or two measurements of the same cell at
 * different times stops being a difference the network has to memorise. It is NOT valid for
 * regression, where the labels are the physical parameter values the scaling changes.
 *
 * The reference is computed from the points the network actually sees, i.e. after any canonical grid
 * resampling, so a client reproducing the transform must do the resampling first.
 */
enum class SpectraNorm
{
	None,
	ZMean,   // divide by the arithmetic mean of |Z|
	ZGeom,   // divide by the geometric mean of |Z|, the centre of the data on a log scale
	ZMax     // divide by the largest |Z|
};

/// Parse a mode name as it appears on the command line and in a checkpoint's meta.json. Returns
/// SpectraNorm::None for an empty or unknown string.
SpectraNorm spectraNormFromString(const std::string& in);
std::string spectraNormToString(SpectraNorm mode);

/// The reference impedance of one spectra, or 0 when the spectra carries no usable magnitude and
/// must be passed through unscaled.
fvalue spectraNormReference(const std::vector<eis::DataPoint>& data, SpectraNorm mode);

/// Divide every point by the reference of the set, leaving the frequencies alone. No-op for
/// SpectraNorm::None, an empty set or a reference of zero.
void applySpectraNorm(std::vector<eis::DataPoint>& data, SpectraNorm mode);

class EisSpectraDataset
{
	inline static const std::string extraInputStr = "exip_";

	CanonicalGrid canonical;
	SpectraNorm spectraNorm = SpectraNorm::None;

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

	/**
	 * Scale every handed out spectra to a common impedance level. Deterministic, so the training and
	 * the test dataset have to be configured identically; the mode is recorded in the network
	 * metadata so _test and libkissinference can rebuild it. Only the classification loaders are
	 * ever given a mode other than SpectraNorm::None.
	 */
	void setSpectraNorm(SpectraNorm mode);
	SpectraNorm spectraNormMode() const;

protected:
	static std::pair<std::vector<std::string>, std::vector<std::string>> getExtraInputsAndLabelNames(const eis::Spectra& spectra);
	virtual eis::Spectra loadSpectraAtIndex(size_t index) = 0;
	virtual eis::Spectra loadSpectraHeaderAtIndex(size_t index) = 0;
	virtual std::vector<std::pair<std::string, int64_t>> extraInputs();
	torch::Tensor frequencies();

	const CanonicalGrid& canonicalGrid() const;
	bool canonicalEnabled() const;

	/**
	 * Turn a raw loaded spectra into the points that go into the input vector: resample onto the
	 * canonical grid when enabled (with the random window crop and frequency shift augmentations
	 * during training), then apply the per spectra |Z| normalization. Returns the input unchanged
	 * when neither is configured, which is the path the regression loaders take.
	 */
	std::vector<eis::DataPoint> resampleCanonical(const std::vector<eis::DataPoint>& data);

private:
	bool canonicalTraining = false;
};
