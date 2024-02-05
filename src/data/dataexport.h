#pragma once
#include <filesystem>
#include <fstream>
#include <eisgenerator/eistype.h>

#include "eistotorch.h"
#include "log.h"
#include "eisdataset.h"

template <typename DatasetType>
bool saveDataset(std::string directory, EisDataset<DatasetType>* dataset)
{
	if(!std::filesystem::is_directory(directory))
	{
		if(!std::filesystem::create_directory(directory))
		{
			Log(Log::ERROR)<<directory<<" is not a directory and a directory could not be created at this path";
			return false;
		}
	}

	size_t datasetSize = 0;
	if(dataset->size().has_value())
	{
		datasetSize = dataset->size().value();
	}
	else
	{
		Log(Log::ERROR)<<"dataset "<<dataset<<" dose not have defined size\n";
		return false;
	}

	for(size_t i = 0; i < datasetSize; ++i)
	{
		torch::data::Example<torch::Tensor, torch::Tensor> example = dataset->get(i);
		std::vector<eis::DataPoint> eisdata = torchToEis(example.data);
		int64_t classNum = example.target.item().to<int64_t>();
		eis::EisSpectra spectra(eisdata, dataset->modelStringForClass(classNum), "");
		spectra.saveToDisk(directory + "/" + std::to_string(classNum) + "_" + std::to_string(i) + ".csv");
	}
	return true;
}
