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

#include <torch/types.h>

#include "log.h"
#include "../eistotorch.h"
#include "progressbar.h"

using namespace eis;

static void purgeEisParamBrackets(std::string& in)
{
	int bracketCounter = 0;

	std::string out;

	for(size_t i = 0; i < in.size(); ++i)
	{
		if(in[i] == '{' || in[i] == '[' || in[i] == '<')
			++bracketCounter;
		if(bracketCounter == 0)
			out.push_back(in[i]);
		if(in[i] == '}' || in[i] == ']' || in[i] == '>')
			--bracketCounter;
	}
	in = out;
}

EisTarDataset::EisTarDataset(const std::filesystem::path& path)
{
	loadTar(path);

	Progress bar("Loading classes: ", files->size()/100);

	for(size_t i = 0; i < files->size(); ++i)
	{
		if(i % 100 == 0)
			bar.tick();
		eis::Spectra spectra = loadSpectraHeaderAtIndex(i);
		generatingModels.push_back(spectra.model);
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

	bar.markAsCompleted();
}

torch::data::Example<torch::Tensor, torch::Tensor> EisTarDataset::getImpl(size_t index)
{
	eis::Spectra data = loadSpectraAtIndex(index);

	std::vector<eis::DataPoint> resampled = resampleCanonical(data.data);
	std::vector<std::string> extraInputNames = getExtraInputsAndLabelNames(data).first;

	torch::Tensor input;
	if(extraInputNames.empty())
	{
		input = eisToTorch(resampled);
	}
	else
	{
		std::vector<fvalue> extraInputs;
		for(const std::string& name : extraInputNames)
			extraInputs.push_back(data.getLabel(name));
		input = eisToTorchExtra(resampled, extraInputs);
	}

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

torch::Tensor EisTarDataset::buildTarget(size_t index) const
{
	if(labelWeights.empty())
	{
		torch::TensorOptions options;
		options = options.layout(torch::kStrided);
		options = options.device(torch::kCPU);
		options = options.dtype(torch::kInt64);
		torch::Tensor output = torch::zeros({1}, options);
		output[0] = static_cast<float>(classIndexes[index]);
		return output;
	}

	std::vector<float> row;
	row.reserve(labelWeights[index].size()+1);
	row.push_back(static_cast<float>(classIndexes[index]));
	row.insert(row.end(), labelWeights[index].begin(), labelWeights[index].end());
	return torch::tensor(row);
}

torch::Tensor EisTarDataset::getTargetImpl(size_t index)
{
	return buildTarget(index);
}

void EisTarDataset::enableSimplicityLabels(const std::vector<double>& omegas, double simplicityPreference,
		double contributionThreshold)
{
	circuit::SimplicityOptions options;
	options.contributionThreshold = contributionThreshold;
	labelWeights = circuit::computeLabelWeights(modelStrs, generatingModels, classIndexes, omegas,
			simplicityPreference, options);
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
