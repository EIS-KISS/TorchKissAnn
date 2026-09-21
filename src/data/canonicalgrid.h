/* * TorchKissAnn - A collection of tools to train various types of Machine learning
 * algorithms on various EIS data
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
#include <cstddef>
#include <vector>
#include <kisstype/type.h>

/**
 * Description of a fixed canonical frequency grid.
 *
 * When enabled, every spectra handed out by a loader is resampled onto this grid before it is
 * flattened into the network input vector. The grid is log spaced by default because EIS frequency
 * axes are log spaced and the impedance of an ECM varies smoothly in log(omega). Resampling makes
 * the position of a feature in the input vector mean the same absolute frequency for every sample,
 * so a network trained on it recognizes a feature irrespective of the frequency range a spectra was
 * measured over, as long as the feature falls inside the grid.
 *
 * Points outside the measured range of an individual spectra are filled by constant extrapolation
 * (see eis::fitToFrequencies), which is the same thing that happens at inference time when a
 * narrower measurement is resampled onto the grid.
 */
struct CanonicalGrid
{
	bool enabled = false;
	double fMin = 0;            // lower edge of the grid, rad/s
	double fMax = 0;            // upper edge of the grid, rad/s
	size_t points = 0;          // number of grid points; 0 means "derive from the first spectra"
	bool log = true;            // log10 spaced when true, linear otherwise

	// Frequency shift augmentation. Shifting a spectra along the log frequency axis is equivalent to
	// scaling every time constant of the circuit by the same factor, which keeps the circuit class
	// unchanged, so it is a valid augmentation for classification. It is NOT valid for regression,
	// where the parameter labels are tied to the absolute position of the features.
	bool augmentShift = false;
	double augmentMaxShift = 1.0; // maximum shift in decades, drawn uniformly from [-max, max]

	// Window (crop) augmentation. A random contiguous sub-range of the measured points is kept and
	// resampled onto the grid as usual, so the parts of the grid outside the crop are filled by
	// constant extrapolation. This simulates a spectra that was only measured over a narrower
	// frequency range. Unlike the shift it does not move features along the axis, so the surviving
	// features keep their absolute position and the parameter labels stay valid: it is safe for
	// regression as well as classification.
	bool augmentWindow = false;
	double augmentWindowMin = 0.5; // minimum fraction of the measured points to keep, in (0, 1]

	// The resolved grid frequencies, size == points. Filled in by the loader once points is known.
	std::vector<fvalue> omegas;

	bool sane() const
	{
		return enabled && fMin > 0 && fMax > fMin && points >= 2;
	}
};
