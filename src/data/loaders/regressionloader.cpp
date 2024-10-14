#include "regressionloader.h"

#include <algorithm>
#include <cassert>
#include <kisstype/type.h>
#include <sstream>
#include <string>

#include "data/eisdataset.h"
#include "microtar.h"
#include "../eistotorch.h"
#include "tensoroptions.h"
#include "tokenize.h"

RegressionLoader::RegressionLoader(const std::filesystem::path& pathI): path(pathI)
{
	int ret = mtar_open(&tar, path.c_str(), "r");
	if(ret != 0)
	{
		Log(Log::WARN)<<path<<" is not a valid tar archive";
		return;
	}

	mtar_header_t header;
	std::vector<std::string> labels;
	size_t length = 0;
	size_t outputCount = 0;
	while((mtar_read_header(&tar, &header)) != MTAR_ENULLRECORD)
	{
		if(header.type == MTAR_TREG)
		{
			size_t pos = tar.pos;
			eis::Spectra spectra = loadSpectraAtCurrentPos(header.size);
			if(labels.empty())
				labels = spectra.labelNames;
			else if(!std::equal(labels.begin(), labels.end(), spectra.labelNames.begin(), spectra.labelNames.end()))
				throw dataset_error(std::string("Not all spectra in ") + path.string() + "have the same labels");

			if(length == 0)
				length = spectra.data.size();
			else if(length != spectra.data.size())
				throw dataset_error(std::string("Not all spectra have the same number of datapoints"));

			if(outputCount == 0)
			{
				std::vector<std::string> labelNames = getExtraInputsAndLabelNames(spectra).second;
				outputCount = labelNames.size();
			}

			files.push_back({.path =  header.name, .pos = pos, .size = header.size});
		}
		mtar_next(&tar);
	}
	if(files.size() > 0)
		getImpl(0);
}

RegressionLoader::RegressionLoader(const RegressionLoader& in)
{
	operator=(in);
}

RegressionLoader& RegressionLoader::operator=(const RegressionLoader& in)
{
	files = in.files;
	path = in.path;
	int ret = mtar_open(&tar, path.c_str(), "r");
	if(ret != 0)
	{
		Log(Log::ERROR)<<"Unable to reopen tar file at "<<path;
		assert(ret == 0);
	}

	return *this;
}

RegressionLoader::~RegressionLoader()
{
	mtar_close(&tar);
}

eis::Spectra RegressionLoader::loadSpectraAtCurrentPos(size_t size)
{
	char* filebuffer = new char[size+1];
	filebuffer[size] = '\0';
	int ret = mtar_read_data(&tar, filebuffer, size);
	if(ret != 0)
	{
		Log(Log::ERROR)<<"Unable to read from tar archive";
		assert(ret == 0);
	}
	std::stringstream ss(filebuffer);

	eis::Spectra spectra = eis::Spectra::loadFromStream(ss);
	delete[] filebuffer;

	return spectra;
}

std::pair<std::vector<std::string>, std::vector<std::string>> RegressionLoader::getExtraInputsAndLabelNames(const eis::Spectra& spectra)
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

std::vector<std::pair<std::string, int64_t>> RegressionLoader::extraInputs()
{
	mtar_seek(&tar, files[0].pos);
	std::vector<std::pair<std::string, int64_t>> extraInputs;
	for(const std::string& name : getExtraInputsAndLabelNames(loadSpectraAtCurrentPos(files[0].size)).first)
	{
		extraInputs.push_back({name, 1});
		extraInputs.back().first.erase(extraInputs.back().first.begin(), extraInputs.back().first.begin()+extraInputStr.length());
	}
	return extraInputs;
}

torch::data::Example<torch::Tensor, torch::Tensor> RegressionLoader::getImpl(size_t index)
{
	if(index >= files.size())
		throw dataset_error("index " + std::to_string(index) + "is out of range for dataset");
	mtar_seek(&tar, files[index].pos);
	eis::Spectra spectra = loadSpectraAtCurrentPos(files[index].size);
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

size_t RegressionLoader::outputSize() const
{
	return outputCount;
}

c10::optional<size_t> RegressionLoader::size() const
{
	return files.size();
}

std::string RegressionLoader::outputName(size_t output)
{
	mtar_seek(&tar, files[0].pos);
	std::vector<std::string> labelNames = getExtraInputsAndLabelNames(loadSpectraAtCurrentPos(files[0].size)).second;
	return labelNames[output];
}

bool RegressionLoader::isMulticlass()
{
	return true;
}

c10::optional<torch::Tensor> RegressionLoader::freqRange()
{
	mtar_seek(&tar, files[0].pos);
	eis::Spectra spectra = loadSpectraAtCurrentPos(files[0].size);
	fvalue front = spectra.data.front().omega;
	fvalue back = spectra.data.back().omega;

	torch::Tensor out = torch::zeros({3}, tensorOptCpu<fvalue>(false));

	out[0] = std::min(front, back);
	out[1] = std::max(front, back);
	out[2] = static_cast<fvalue>(spectra.data.size());
	return out;
}
