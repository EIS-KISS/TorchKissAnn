#include "dirloader.h"

#include <algorithm>
#include <assert.h>
#include <eisgenerator/eistype.h>
#include <eisgenerator/translators.h>

#include "model.h"
#include "mpt.h"
#include "../print.h"
#include "log.h"
#include "../../utils/tokenize.h"
#include "../filterdata.h"
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

	for(const std::filesystem::directory_entry& dirent : std::filesystem::directory_iterator{directoryPath})
	{
		if(!dirent.is_regular_file() || dirent.path().extension() != ".csv")
			continue;
		Log(Log::DEBUG)<<"Using: "<<dirent.path().filename();
		eis::EisSpectra spectra = eis::EisSpectra::loadFromDisk(dirent.path());
		purgeEisParamBrackets(spectra.model);
		eis::Model::removeSeriesResitance(spectra.model);
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

	eis::EisSpectra data;

	try
	{
		data = eis::EisSpectra::loadFromDisk(fileNames[index].path);
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

	filterData(data.data, inputSize);

	torch::Tensor input = eisToTorch(data.data);
	return torch::data::Example<torch::Tensor, torch::Tensor>(input, output);
}

size_t EisDirDataset::classForIndex(size_t index)
{
	return fileNames[index].classNum;
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

std::string EisDirDataset::modelStringForClass(size_t classNum)
{
	if(classNum >= modelStrs.size())
		return "invalid";
	else
		return *std::next(modelStrs.begin(), classNum);
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
