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

class EisDirDataset : public DirDataset, public EisDataset<EisDirDataset>
{
private:
	std::vector<std::string> modelStrs;
	std::vector<size_t> classIndexes;

	virtual torch::data::Example<torch::Tensor, torch::Tensor> getImpl(size_t index) override;

public:
	explicit EisDirDataset(const std::filesystem::path& path);
	EisDirDataset(const EisDirDataset& in) = default;

	torch::Tensor modelWeights();

	virtual c10::optional<size_t> size() const override;

	virtual size_t outputSize() const override;
	virtual std::string outputName(size_t output) override;
	virtual torch::Tensor classCounts() override;
	virtual std::vector<std::pair<std::string, int64_t>> extraInputs() override;
	virtual c10::optional<torch::Tensor> frequencies() override;
};
