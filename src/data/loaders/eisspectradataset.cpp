//
// TorchKissAnn - A collection of tools to train various types of Machine learning
// algorithms on various types of EIS data
// Copyright (C) 2025 Carl Klemm <carl@uvos.xyz>
//
// This file is part of TorchKissAnn.
//
// TorchKissAnn is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// TorchKissAnn is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with TorchKissAnn.  If not, see <http://www.gnu.org/licenses/>.
//

#include "eisspectradataset.h"
#include <algorithm>
#include <cmath>
#include <eisgenerator/basicmath.h>

#include "data/eistotorch.h"
#include "randomgen.h"

std::pair<std::vector<std::string>, std::vector<std::string>> EisSpectraDataset::getExtraInputsAndLabelNames(const eis::Spectra& spectra)
{
	std::vector<std::string> labelNames;
	std::vector<std::string> extraInputNames;
	for(const std::string& label : spectra.labelNames)
	{
		if(label.find(extraInputStr) == 0)
			extraInputNames.push_back(label);
		else
			labelNames.push_back(label);
	}

	return {extraInputNames, labelNames};
}

std::vector<std::pair<std::string, int64_t>> EisSpectraDataset::extraInputs()
{
	eis::Spectra spectra = loadSpectraAtIndex(0);
	std::vector<std::pair<std::string, int64_t>> out;
	for(const std::string& name : getExtraInputsAndLabelNames(spectra).first)
	{
		out.push_back({name, 1});
		out.back().first.erase(out.back().first.begin(), out.back().first.begin()+extraInputStr.length());
	}
	return out;
}

torch::Tensor EisSpectraDataset::frequencies()
{
	if(canonical.enabled)
	{
		// The input vector holds the real part of every grid point followed by the imaginary part,
		// so the frequency of index i is the grid frequency i % points, matching eisToTorch().
		std::vector<fvalue> omega(canonical.points*2);
		for(size_t i = 0; i < omega.size(); ++i)
			omega[i] = canonical.omegas[i % canonical.points];
		return fvalueVectorToTensor(omega);
	}

	eis::Spectra spectra = loadSpectraAtIndex(0);
	std::vector<fvalue> omega(spectra.data.size());
	for(size_t i = 0; i < spectra.data.size(); ++i)
		omega[i] = spectra.data[i].omega;
	return fvalueVectorToTensor(omega);
}

void EisSpectraDataset::setCanonicalGrid(const CanonicalGrid& grid)
{
	canonical = grid;
	if(!canonical.enabled)
		return;

	if(canonical.points == 0)
		canonical.points = loadSpectraAtIndex(0).data.size();

	eis::Range range(canonical.fMin, canonical.fMax, canonical.points, canonical.log);
	canonical.omegas = range.getRangeVector();
}

const CanonicalGrid& EisSpectraDataset::canonicalGrid() const
{
	return canonical;
}

bool EisSpectraDataset::canonicalEnabled() const
{
	return canonical.enabled;
}

void EisSpectraDataset::setCanonicalTraining(bool training)
{
	canonicalTraining = training;
}

std::vector<eis::DataPoint> EisSpectraDataset::resampleCanonical(const std::vector<eis::DataPoint>& data)
{
	if(!canonical.enabled)
		return data;

	std::vector<fvalue> grid = canonical.omegas;

	std::vector<eis::DataPoint> sorted = data;
	std::sort(sorted.begin(), sorted.end());

	// Window augmentation: keep a random contiguous slice of the measured points. The grid is
	// resampled from this slice as usual, so grid points outside the slice fall back to the value at
	// the nearest slice edge (constant extrapolation), mimicking a narrower measurement.
	if(canonical.augmentWindow && canonicalTraining && canonical.augmentWindowMin < 1.0 && sorted.size() >= 2)
	{
		size_t n = sorted.size();
		size_t minLen = (size_t)std::ceil(canonical.augmentWindowMin * (double)n);
		if(minLen < 2)
			minLen = 2;
		if(minLen < n)
		{
			size_t maxStart = n - minLen;
			size_t start = (size_t)(rd::rand() * (double)(maxStart + 1));
			if(start > maxStart)
				start = maxStart;
			// Length between minLen and whatever remains from start to the end of the data.
			size_t extra = n - start - minLen;
			size_t len = minLen + (extra > 0 ? (size_t)(rd::rand() * (double)(extra + 1)) : 0);
			if(len > n - start)
				len = n - start;
			sorted = std::vector<eis::DataPoint>(sorted.begin() + start, sorted.begin() + start + len);
		}
	}

	// Shifting the query grid by a factor of 10^delta evaluates Z(omega / 10^delta), which is the
	// original spectra moved delta decades along the frequency axis. This scales every time constant
	// of the circuit by the same factor and therefore preserves the circuit class.
	if(canonical.augmentShift && canonicalTraining && canonical.augmentMaxShift > 0)
	{
		double delta = rd::rand(-canonical.augmentMaxShift, canonical.augmentMaxShift);
		double factor = std::pow(10.0, delta);
		for(fvalue& omega : grid)
			omega = static_cast<fvalue>(omega / factor);
	}

	return eis::fitToFrequencies(grid, sorted, false);
}
