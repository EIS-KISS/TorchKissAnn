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
#include <filesystem>
#include <kisstype/spectra.h>

#include "data/regressiondataset.h"
#include "data/loaders/tardataset.h"

class RegressionLoaderTar :
public TarDataset, public RegressionDataset<RegressionLoaderTar>
{
protected:
	virtual torch::data::Example<torch::Tensor, torch::Tensor> getImpl(size_t index) override;

	size_t outputCount;

public:
	explicit RegressionLoaderTar(const std::filesystem::path& pathI);

	virtual size_t outputSize() const override;
	virtual std::string outputName(size_t output) override;
	virtual c10::optional<size_t> size() const override;
	virtual bool isMulticlass() override;
	virtual std::vector<std::pair<std::string, int64_t>> extraInputs() override;
	virtual const std::string targetName() override;
	virtual c10::optional<torch::Tensor> frequencies() override;
};
