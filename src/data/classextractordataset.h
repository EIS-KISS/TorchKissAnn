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
	_dataset(dataset), datasetMutex(new std::mutex), _classNum(classNum)
	{
		size_t dataSetSize = _dataset->size().value();
		for(size_t i = 0; i < dataSetSize; ++i)
		{
			torch::Tensor target = _dataset->getTarget(i);
			if(target.item().toUInt64() == classNum)
				indexList.push_back(i);
		}
	}

	virtual size_t outputSize() const override
	{
		return 1;
	}

	virtual std::string outputName(size_t classNum) override
	{
		return _dataset->outputName(_classNum);
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
