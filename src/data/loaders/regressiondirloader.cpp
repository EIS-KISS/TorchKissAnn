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

#include "regressiondirloader.h"

#include <algorithm>
#include <cassert>
#include <kisstype/type.h>
#include <string>

#include "microtar.h"
#include "../eistotorch.h"
#include "tensoroptions.h"

RegressionLoaderDir::RegressionLoaderDir(const std::filesystem::path& pathI)
{
	loadDir(pathI);

	outputCount = getExtraInputsAndLabelNames(loadSpectraAtIndex(0)).second.size();

	if(files->size() > 0)
		getImpl(0);
}

torch::data::Example<torch::Tensor, torch::Tensor> RegressionLoaderDir::getImpl(size_t index)
{
	eis::Spectra spectra = loadSpectraAtIndex(index);
	auto pair = getExtraInputsAndLabelNames(spectra);
	std::vector<std::string> labelNames = pair.second;
	std::vector<std::string> extraInputNames = pair.first;

	torch::Tensor input;
	if(extraInputNames.empty())
	{
		 input = eisToTorch(spectra.data);
	}
	else
	{
		std::vector<fvalue> extraInputs;
		for(const std::string& name : labelNames)
			extraInputs.push_back(spectra.getLabel(name));
		input = eisToTorchExtra(spectra.data, extraInputs);
	}

	torch::TensorOptions options;
	options = options.layout(torch::kStrided);
	options = options.device(torch::kCPU);
	options = options.dtype(torch::kFloat32);
	torch::Tensor output = torch::empty({static_cast<int64_t>(labelNames.size())}, options);

	size_t i = 0;
	auto tensorAccessor = output.accessor<float, 1>();
	for(const std::string& labelNames : labelNames)
	{
		tensorAccessor[i] = spectra.getLabel(labelNames);
		++i;
	}

	return torch::data::Example<torch::Tensor, torch::Tensor>(input, output);
}

size_t RegressionLoaderDir::outputSize() const
{
	return outputCount;
}

c10::optional<size_t> RegressionLoaderDir::size() const
{
	return files->size();
}

std::string RegressionLoaderDir::outputName(size_t output)
{
	std::vector<std::string> labelNames = getExtraInputsAndLabelNames(loadSpectraAtIndex(0)).second;
	return labelNames.at(output);
}

bool RegressionLoaderDir::isMulticlass()
{
	return true;
}

c10::optional<torch::Tensor> RegressionLoaderDir::frequencies()
{
	return EisSpectraDataset::frequencies();
}

std::vector<std::pair<std::string, int64_t>> RegressionLoaderDir::extraInputs()
{
	return DirDataset::extraInputs();
}

const std::string RegressionLoaderDir::targetName()
{
	eis::Spectra spectra = loadSpectraAtIndex(0);
	if(spectra.model.empty())
		return "Unkown";
	return spectra.model;
}
