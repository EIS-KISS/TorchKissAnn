#pragma once
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <optional>
#include <limits>
#include <memory>
#include <filesystem>
#include <fstream>

#include "constants.h"
#include "classextractordatasethighspeed.h"
#include "log.h"
#include "../csvBackend.h"
#include "tokenize.h"

template <typename ParentType>
class DeduplatorDataset : public EisDataset<DeduplatorDataset>
{
	static constexpr size_t BATCH_SIZE = 100000;
	static constexpr float epsilon = 0.001;
private:
	EisDataset<ParentType>* _dataset;
	std::vector<uint8_t> originClass;
	std::vector<torch::Tensor> targets;
	bool ready = false;

public:

	DeduplatorDataset(EisDataset<ParentType>* dataset, const std::filesystem::path& fileName):
	_dataset(dataset)
	{
		io::LineReader dedupLine(fileName);
		std::string firstLineStr(dedupLine.next_line());
		std::vector<std::string> firstLineTokens = tokenize(firstLineStr, ',');

		if(firstLineTokens[0].find("id") == std::string::npos)
		{
			Log(Log::ERROR)<<"file "<<fileName<<" dosent contain a valid csv deduplication header";
			return;
		}

		bool hasOrigin = (firstLineTokens.size() > 1 && firstLineTokens[1].find("originClass") != std::string::npos);
		if(firstLineTokens.size()-1-hasOrigin != _dataset->classesCount())
		{
			Log(Log::ERROR)<<"file "<<fileName<<" is not a deduplication file for the current dataset";
			return;
		}

		torch::TensorOptions options;
		options = options.dtype(torch::kInt8);
		options = options.layout(torch::kStrided);
		options = options.device(torch::kCPU);
		size_t datasetSize = dataset->size().value();
		targets.resize(datasetSize);
		for(size_t i = 0; i < datasetSize; ++i)
			targets[i] = torch::zeros({static_cast<int64_t>(_dataset->classesCount())}, options);

		try
		{
			while(char* line = dedupLine.next_line())
			{
				std::string lineStr(line);
				std::vector<std::string> tokens = tokenize(lineStr, ',');
				if(tokens.size()-1-hasOrigin != _dataset->classesCount())
					return;
				if(hasOrigin)
					originClass.push_back(std::stoul(tokens[1]));

				size_t id = std::stoul(tokens[0]);
				if(id > targets.size()-1)
				{
					Log(Log::WARN)<<"id "<<id<<" is out of range for dataset of size "<<datasetSize;
					continue;
				}

				for(size_t i = 0; i < _dataset->classesCount(); ++i)
					targets[id][i] = std::stoul(tokens[1]);
			}

			ready = true;
		}
		catch(const std::invalid_argument& ex)
		{
			Log(Log::ERROR)<<"file "<<fileName<<" contains invalid numbers";
		}
	}

	DeduplatorDataset(EisDataset<ParentType>* dataset) : _dataset(dataset)
	{
		size_t dataSetSize = dataset->size().value();
		targets.resize(dataSetSize);
		originClass.resize(dataSetSize, std::numeric_limits<uint8_t>::max());

		std::vector<std::unique_ptr<ClassExtractorDatasetHighSpeed<ParentType>>> classExtractorDatasets;

		for(size_t i = 0; i < _dataset->classesCount(); ++i)
		{
			classExtractorDatasets.push_back(
				std::unique_ptr<ClassExtractorDatasetHighSpeed<ParentType>>(new ClassExtractorDatasetHighSpeed<ParentType>(dataset, i)));
		}

		torch::data::DataLoaderOptions loaderOptions;
		loaderOptions = loaderOptions.batch_size(BATCH_SIZE);
		loaderOptions = loaderOptions.workers(JOBS);
		loaderOptions = loaderOptions.max_jobs(JOBS*2);

		std::vector<std::unique_ptr<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<ClassExtractorDatasetHighSpeed<ParentType>,
		torch::data::transforms::Stack<torch::data::Example<>> >,
		torch::data::samplers::SequentialSampler> > > classLoaders;
		//yes i know wtf

		for(size_t i = 0; i < _dataset->classesCount(); ++i)
		{
			loaderOptions = loaderOptions.batch_size(std::min(classExtractorDatasets[i]->size().value()/(JOBS*2), BATCH_SIZE));
			classLoaders.push_back(torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
				classExtractorDatasets[i]->map(torch::data::transforms::Stack<>()), loaderOptions));
		}

		torch::TensorOptions options;
		options = options.dtype(torch::kInt8);
		options = options.layout(torch::kStrided);
		options = options.device(torch::kCPU);

		for(size_t i = 0; i < dataSetSize; ++i)
			targets[i] = torch::zeros({static_cast<int64_t>(_dataset->classesCount())}, options);

		for(size_t classNum = 0; classNum < _dataset->classesCount()-1; ++classNum)
		{
			Log(Log::INFO)<<"Starting for Class "<<classNum;
			size_t classSize = classExtractorDatasets[classNum]->size().value();
			for(size_t i = 0; i < classSize; ++i)
			{
				auto masterExample = classExtractorDatasets[classNum]->get(i);
				size_t parentIndex = classExtractorDatasets[classNum]->getParentIndex(i);
				targets[parentIndex][masterExample.target.template item<int8_t>()] = 1;

				float smalest = std::numeric_limits<float>::max();
				for(size_t testClassNum = classNum+1; testClassNum < _dataset->classesCount(); ++testClassNum)
				{
					size_t batchStart = 0;
					for(auto& batch : *classLoaders[testClassNum])
					{
						torch::Tensor master = masterExample.data.repeat({batch.data.size(0), 1});
						master = (master-batch.data).pow(2).sum(1, true).div(masterExample.data.size(0));
						float* masterDataPtr = master.contiguous().template data_ptr<float>();
						for(size_t j = 0; j < static_cast<size_t>(master.size(0)); ++j)
						{
							if(batch.target[j].template item<int8_t>() != masterExample.target.template item<int8_t>())
							{
								if(masterDataPtr[j] < smalest)
									smalest = masterDataPtr[j];
								if(masterDataPtr[j] < epsilon)
								{
									targets[parentIndex][batch.target[j].template item<int8_t>()] = 1;
									size_t parentTestNum = classExtractorDatasets[testClassNum]->getParentIndex(batchStart+j);
									targets[parentTestNum][masterExample.target.template item<int8_t>()] = 1;
								}
							}
						}
						batchStart += batch.data.size(0);
					}
				}
				Log(Log::INFO)<<"Class "<<classNum<<" of "<<_dataset->classesCount()-1<<'\t'<<i<<'\\'<<classSize<<"\tsmallest match: "<<smalest;
			}
			ready = true;
		}

		size_t classSize = classExtractorDatasets[_dataset->classesCount()-1]->size().value();
		for(size_t i = 0; i < classSize; ++i)
		{
			Log(Log::INFO)<<"Class "<<_dataset->classesCount()-1<<" of "<<_dataset->classesCount()-1<<'\t'<<i<<'\\'<<classSize;
			auto masterExample = classExtractorDatasets[_dataset->classesCount()-1]->get(i);
			size_t parentIndex = classExtractorDatasets[_dataset->classesCount()-1]->getParentIndex(i);
			targets[parentIndex][masterExample.target.template item<int8_t>()] = 1;
		}
	}

	torch::data::Example<torch::Tensor, torch::Tensor> get(size_t index) override
	{
		assert(index < targets.size());

		torch::data::Example<torch::Tensor, torch::Tensor> example = _dataset->get(index);

		return torch::data::Example<torch::Tensor, torch::Tensor>(example.data, targets[index]);
	}

	bool save(const std::filesystem::path& path)
	{
		std::fstream file;
		file.open(path, std::fstream::out | std::fstream::trunc);
		if(!file.is_open())
		{
			Log(Log::WARN)<<path<<" can not be opened for writeing";
			return false;
		}

		file<<"id, originClass, ";
		for(size_t i = 0; i < _dataset->classesCount(); ++i)
		{
			file<<"class"<<i;
			if(i+1 < _dataset->classesCount())
				file<<", ";
		}
		file<<"\n ";

		for(size_t i = 0; i < targets.size(); ++i)
		{
			file<<i<<", ";
			for(size_t j = 0; j < _dataset->classesCount(); ++j)
			{
				file<<targets[i][j].item().to<int>();
				if(j+1 < _dataset->classesCount())
					file<<", ";
			}
			file<<'\n';
		}

		file.close();
		return true;
	}

	virtual size_t classesCount() const override
	{
		return _dataset->classesCount();
	}

	virtual c10::optional<size_t> size() const override
	{
		return targets.size();
	}

	virtual size_t classForIndex(size_t index) override
	{
		return originClass[index];
	}

	bool isReady()
	{
		return ready;
	}
};
