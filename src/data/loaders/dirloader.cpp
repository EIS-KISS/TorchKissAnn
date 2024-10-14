#include "dirloader.h"

#include <algorithm>
#include <assert.h>
#include <kisstype/spectra.h>
#include <kisstype/type.h>

#include <eisgenerator/translators.h>

#include "log.h"
#include "../eistotorch.h"

using namespace eis;

EisDirDataset::EisDirDataset(const std::string& dirName, int64_t inputSizeI):
inputSize(inputSizeI)
{
	const std::filesystem::path directoryPath{dirName};
	if(!std::filesystem::is_directory(dirName))
	{
		Log(Log::WARN)<<dirName<<" is not a valid directory";
		return;
	}

	size_t dataSize = 0;
	for(const std::filesystem::directory_entry& dirent : std::filesystem::directory_iterator{directoryPath})
	{
		if(!dirent.is_regular_file() || dirent.path().extension() != ".csv")
			continue;
		Log(Log::DEBUG)<<"Using: "<<dirent.path().filename();
		eis::Spectra spectra = eis::Spectra::loadFromDisk(dirent.path());
		purgeEisParamBrackets(spectra.model);

		if(dataSize == 0)
		{
			dataSize = spectra.data.size();
		}
		else if(dataSize != spectra.data.size())
		{
			Log(Log::WARN)<<"Spectra at "<<dirent.path()<<" has length "<<spectra.data.size()<<" but expected "<<dataSize;
			continue;
		}

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
		fileNames.push_back({dirent.path(), index});
	}
	if(fileNames.size() < 20)
		Log(Log::WARN)<<"found few valid files in "<<directoryPath;
}

torch::data::Example<torch::Tensor, torch::Tensor> EisDirDataset::getImpl(size_t index)
{
	if(fileNames.size() < index)
	{
		Log(Log::ERROR)<<"index "<<index<<" out of range in "<<__func__;
		assert(false);
	}

	torch::TensorOptions options;
	options = options.layout(torch::kStrided);
	options = options.device(torch::kCPU);
	options = options.dtype(torch::kInt64);
	torch::Tensor output = torch::zeros({1}, options);
	output[0] = static_cast<float>(fileNames[index].classNum);

	eis::Spectra data;

	try
	{
		data = eis::Spectra::loadFromDisk(fileNames[index].path);
	}
	catch(const eis::file_error& err)
	{
		Log(Log::WARN)<<"Can't load datafile from "<<fileNames[index].path<<' '<<err.what();
		if(index != 0)
		{
			if(index+1 < size().value())
				return get(index+1);
			else
				return get(0);
		}
		assert(false);
	}

	torch::Tensor input = eisToTorch(data.data);
	return torch::data::Example<torch::Tensor, torch::Tensor>(input, output);
}

size_t EisDirDataset::outputSize() const
{
	size_t classes = 0;
	for(const FileNameStr& str : fileNames)
	{
		if(str.classNum+1 > classes)
			classes = str.classNum+1;
	}

	return classes;
}

c10::optional<size_t> EisDirDataset::size() const
{
	return fileNames.size();
}

torch::Tensor EisDirDataset::modelWeights()
{
	size_t classes = outputSize();

	torch::TensorOptions options;
	options = options.dtype(torch::kFloat32);
	options = options.layout(torch::kStrided);
	options = options.device(torch::kCPU);
	torch::Tensor output = torch::empty({static_cast<int64_t>(classes)}, options);

	float* tensorDataPtr = output.contiguous().data_ptr<float>();
	for(const FileNameStr& str : fileNames)
		++tensorDataPtr[str.classNum];

	return output;
}

std::string EisDirDataset::outputName(size_t output)
{
	if(output >= modelStrs.size())
		return "invalid";
	else
		return *std::next(modelStrs.begin(), output);
}

torch::Tensor EisDirDataset::classCounts()
{
	torch::TensorOptions options;
	options = options.dtype(torch::kFloat32);
	options = options.layout(torch::kStrided);
	options = options.device(torch::kCPU);
	torch::Tensor out = torch::zeros({static_cast<int64_t>(outputSize())}, options);

	for(const FileNameStr& file : fileNames)
		out[file.classNum] = out[file.classNum] + 1;
	return out;
}
