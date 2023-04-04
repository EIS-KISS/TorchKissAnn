#pragma once
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <optional>
#include <mutex>
#include <memory>

#include "data/eisdataset.h"

template <typename ParentType>
class ClassExtractorDataset : public EisDataset<ClassExtractorDataset<ParentType>>
{
private:
	EisDataset<ParentType>* _dataset;
	std::shared_ptr<std::mutex> datasetMutex;
	size_t _classNum;

	std::vector<size_t> indexList;

	virtual torch::data::Example<torch::Tensor, torch::Tensor> getImpl(size_t index) override
	{
		assert(index < indexList.size());

		datasetMutex->lock();
		torch::data::Example<torch::Tensor, torch::Tensor> example = _dataset->get(indexList[index]);
		datasetMutex->unlock();
		return example;
	}

public:

	ClassExtractorDataset(EisDataset<ParentType>* dataset, size_t classNum):
	_dataset(dataset), _classNum(classNum), datasetMutex(new std::mutex)
	{
		size_t dataSetSize = _dataset->size().value();
		for(size_t i = 0; i < dataSetSize; ++i)
		{
			torch::data::Example<torch::Tensor, torch::Tensor> example = _dataset->get(i);
			if(example.target.item<int64_t>() == static_cast<int64_t>(classNum))
			{
				indexList.push_back(i);
			}
		}
	}

	virtual size_t classesCount() const override
	{
		return 1;
	}

	virtual size_t classForIndex(size_t index) override
	{
		return _classNum;
	}

	virtual size_t modelStringForClass(size_t classNum)
	{
		return _dataset->modelStringForClass(_classNum);
	}

	size_t getParentIndex(size_t index)
	{
		assert(index < indexList.size());

		return indexList[index];
	}

	virtual c10::optional<size_t> size() const override
	{
		return indexList.size();
	}
};
