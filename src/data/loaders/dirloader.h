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

#include <vector>
#include <string>
#include <filesystem>
#include <kisstype/type.h>

#include "data/loaders/dirdataset.h"
#include "data/eisdataset.h"
#include "loss/circuitsimplicity.h"

class EisDirDataset : public DirDataset, public EisDataset<EisDirDataset>
{
private:
	std::vector<std::string> modelStrs;
	std::vector<size_t> classIndexes;
	/**
	 * The circuit of each spectra exactly as its header spells it, parameters included. The class
	 * table above has the parameters stripped, but telling whether an element of a circuit has
	 * become negligible needs their values.
	 */
	std::vector<std::string> generatingModels;
	/**
	 * Filled by enableSimplicityLabels(): one weight per class per sample. While it is empty the
	 * loader hands out the bare class index as its target, when it is set the target carries the
	 * index followed by those weights.
	 */
	std::vector<std::vector<float>> labelWeights;

	virtual torch::data::Example<torch::Tensor, torch::Tensor> getImpl(size_t index) override;
	virtual torch::Tensor getTargetImpl(size_t index) override;

	torch::Tensor buildTarget(size_t index) const;

public:
	explicit EisDirDataset(const std::filesystem::path& path);
	EisDirDataset(const EisDirDataset& in) = default;

	/**
	 * Hand out, instead of the class index, the set of classes that describe each spectra: the
	 * generating circuit plus every class in this dataset that a circuit missing one of its
	 * elements reduces to. See circuit::computeLabelWeights().
	 *
	 * The target becomes a {1 + classCount} float vector, the class index followed by one weight per
	 * class, which ann::classification::train() and test() read back. Call it on the training and the
	 * test dataset with the same arguments or the two are not scoring the same labels.
	 */
	void enableSimplicityLabels(const std::vector<double>& omegas, double simplicityPreference,
			double contributionThreshold) override;

	bool hasSimplicityLabels() const override
	{
		return !labelWeights.empty();
	}

	torch::Tensor modelWeights();

	virtual c10::optional<size_t> size() const override;

	virtual size_t outputSize() const override;
	virtual std::string outputName(size_t output) override;
	virtual torch::Tensor classCounts() override;
	virtual std::vector<std::pair<std::string, int64_t>> extraInputs() override;
	virtual c10::optional<torch::Tensor> frequencies() override;
};
