#include "dirloader.h"

#include <algorithm>
#include <assert.h>
#include <eisgenerator/eistype.h>

#include "csv.h"
#include "mpt.h"
#include "../print.h"
#include "log.h"
#include "../../utils/tokenize.h"
#include "../filterdata.h"
#include "../eistotorch.h"

using namespace eis;

EisDirDataset::EisDirDataset(const std::string& dirName, const std::string& prefix, FileType fileTypeI, int64_t outputSizeI):
fileType(fileTypeI), outputSize(outputSizeI)
{
	const std::filesystem::path directoryPath{dirName};
	for(const std::filesystem::directory_entry& dirent : std::filesystem::directory_iterator{directoryPath})
	{
		std::string fileName = dirent.path().filename();
		std::cout<<fileName;
		if(dirent.is_regular_file() && fileName.size() > prefix.size() && fileName.find(prefix) == 0)
		{
			fileName.erase(fileName.begin(), fileName.begin()+prefix.size());
			std::vector<std::string> tokens = tokenize(fileName, '_');
			size_t classNum = std::stoul(tokens[0]);
			fileNames.push_back({dirent.path(), classNum});
		}
	}
}

torch::data::Example<torch::Tensor, torch::Tensor> EisDirDataset::get(size_t index)
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

	std::vector<eis::DataPoint> data;

	switch(fileType)
	{
		case FILE_TYPE_CSV:
			data = loaders::csv(fileNames[index].path);
			break;
		case FILE_TYPE_MPT:
			data = loaders::mpt(fileNames[index].path);
			break;
		case FILE_TYPE_RELAXIS:
		case FILE_TYPE_INVALID:
		case FILE_TYPE_TRASH:
		case FILE_TYPE_GAN:
		default:
			Log(Log::ERROR)<<"invalid file type "<<fileType<<" in "<<__func__;
			assert(false);
			break;
	}

	if(data.empty())
	{
		if(index != 0)
		{
			Log(Log::WARN)<<"Can't load datafile from "<<fileNames[index].path;
			if(index+1 < size().value())
				return get(index+1);
			else
				return get(0);
		}
		else
		{
			assert(false);
		}
	}

	filterData(data, outputSize);

	torch::Tensor input = eisToTorch(data);
	return torch::data::Example<torch::Tensor, torch::Tensor>(input, output);
}

size_t EisDirDataset::modelCount()
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
	size_t classes = modelCount();

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
