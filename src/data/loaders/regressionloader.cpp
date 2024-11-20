#include "regressionloader.h"

#include <algorithm>
#include <cassert>
#include <kisstype/type.h>
#include <string>

#include "../eistotorch.h"
#include "tensoroptions.h"

RegressionLoaderTar::RegressionLoaderTar(const std::filesystem::path& pathI)
{
	loadTar(pathI);

	outputCount = getExtraInputsAndLabelNames(loadSpectraAtIndex(0)).second.size();

	if(files->size() > 0)
		getImpl(0);
}

torch::data::Example<torch::Tensor, torch::Tensor> RegressionLoaderTar::getImpl(size_t index)
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

size_t RegressionLoaderTar::outputSize() const
{
	return outputCount;
}

c10::optional<size_t> RegressionLoaderTar::size() const
{
	return files->size();
}

std::string RegressionLoaderTar::outputName(size_t output)
{
	std::vector<std::string> labelNames = getExtraInputsAndLabelNames(loadSpectraAtIndex(0)).second;
	return labelNames[output];
}

bool RegressionLoaderTar::isMulticlass()
{
	return true;
}

c10::optional<torch::Tensor> RegressionLoaderTar::frequencies()
{
	return EisSpectraDataset::frequencies();
}

std::vector<std::pair<std::string, int64_t>> RegressionLoaderTar::extraInputs()
{
	return TarDataset::extraInputs();
}

const std::string RegressionLoaderTar::targetName()
{
	eis::Spectra spectra = loadSpectraAtIndex(0);
	if(spectra.model.empty())
		return "Unkown";
	return spectra.model;
}
