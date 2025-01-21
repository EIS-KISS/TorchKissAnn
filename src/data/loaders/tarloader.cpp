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

#include "tarloader.h"

#include <assert.h>
#include <kisstype/spectra.h>
#include <kisstype/type.h>
#include <algorithm>

#include <eisgenerator/translators.h>
#include <torch/types.h>

#include "log.h"
#include "../eistotorch.h"
#include "indicators.hpp"

using namespace eis;

EisTarDataset::EisTarDataset(const std::filesystem::path& path)
{
	loadTar(path);

	indicators::show_console_cursor(false);

	indicators::BlockProgressBar bar(
		indicators::option::BarWidth(50),
		indicators::option::PrefixText("Loading classes: "),
		indicators::option::ShowElapsedTime(true),
		indicators::option::ShowRemainingTime(true),
		indicators::option::MaxProgress(files->size()/100)
	);

	for(size_t i = 0; i < files->size(); ++i)
	{
		if(i % 100 == 0)
			bar.tick();
		eis::Spectra spectra = loadSpectraHeaderAtIndex(i);
		purgeEisParamBrackets(spectra.model);

		if(spectra.model.length() < 2 && spectra.model != "r" && spectra.model != "c" && spectra.model != "w" && spectra.model != "p" && spectra.model != "l")
			spectra.model = "Union";
		auto search = std::find(modelStrs.begin(), modelStrs.end(), spectra.model);
		size_t index;
		if(search == modelStrs.end())
		{
			index = modelStrs.size();
			modelStrs.push_back(spectra.model);
			Log(Log::DEBUG)<<"New model "<<index<<": "<<spectra.model;
		}
		else
		{
			index = search - modelStrs.begin();
		}
		classIndexes.push_back(index);
	}

	bar.mark_as_completed();
	indicators::show_console_cursor(true);
}

torch::data::Example<torch::Tensor, torch::Tensor> EisTarDataset::getImpl(size_t index)
{
	eis::Spectra data = loadSpectraAtIndex(index);

	torch::Tensor input = eisToTorch(data.data);
	return torch::data::Example<torch::Tensor, torch::Tensor>(input, EisTarDataset::getTargetImpl(index));
}

size_t EisTarDataset::outputSize() const
{
	return *std::max_element(classIndexes.begin(), classIndexes.end()) + 1;
}

c10::optional<size_t> EisTarDataset::size() const
{
	return files->size();
}

torch::Tensor EisTarDataset::modelWeights()
{
	size_t classes = outputSize();

	torch::TensorOptions options;
	options = options.dtype(torch::kFloat32);
	options = options.layout(torch::kStrided);
	options = options.device(torch::kCPU);
	torch::Tensor output = torch::empty({static_cast<int64_t>(classes)}, options);

	float* tensorDataPtr = output.contiguous().data_ptr<float>();
	for(size_t classNum : classIndexes)
		++tensorDataPtr[classNum];

	return output;
}

torch::Tensor EisTarDataset::getTargetImpl(size_t index)
{
	torch::TensorOptions options;
	options = options.layout(torch::kStrided);
	options = options.device(torch::kCPU);
	options = options.dtype(torch::kInt64);
	torch::Tensor output = torch::zeros({1}, options);
	output[0] = static_cast<float>(classIndexes[index]);

	return output;
}

std::string EisTarDataset::outputName(size_t output)
{
	if(output >= modelStrs.size())
		return "invalid";
	else
		return *std::next(modelStrs.begin(), output);
}

c10::optional<torch::Tensor> EisTarDataset::frequencies()
{
	return EisSpectraDataset::frequencies();
}

torch::Tensor EisTarDataset::classCounts()
{
	torch::TensorOptions options;
	options = options.dtype(torch::kInt64);
	options = options.layout(torch::kStrided);
	options = options.device(torch::kCPU);
	torch::Tensor out = torch::zeros({static_cast<int64_t>(outputSize())}, options);
	auto outAcc = out.accessor<int64_t, 1>();

	for(size_t classNum : classIndexes)
		outAcc[classNum] = outAcc[classNum] + 1;
	return out;
}

std::vector<std::pair<std::string, int64_t>> EisTarDataset::extraInputs()
{
	return TarDataset::extraInputs();
}
