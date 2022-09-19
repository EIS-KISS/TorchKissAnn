#pragma once
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <optional>
#include <mutex>

#include "data/eisdataset.h"

template <typename ParentType>
class ClassExtractorDatasetHighSpeed : public EisDataset<ClassExtractorDatasetHighSpeed<ParentType>>
{
private:
	std::vector<torch::data::Example<torch::Tensor, torch::Tensor>> examples;
	std::vector<size_t> indexList;
	size_t _classNum;
	std::string modelString;

public:

	ClassExtractorDatasetHighSpeed(EisDataset<ParentType>* dataset, size_t classNum): _classNum(classNum)
	{
		modelString = dataset->modelStringForClass(_classNum);
		size_t dataSetSize = dataset->size().value();
		for(size_t i = 0; i < dataSetSize; ++i)
		{
			torch::data::Example<torch::Tensor, torch::Tensor> example = dataset->get(i);
			if(example.target.item<int64_t>() == static_cast<int64_t>(classNum))
			{
				examples.push_back(example);
				indexList.push_back(i);
			}
		}
	}

	size_t getParentIndex(size_t index)
	{
		assert(index < examples.size());

		return indexList[index];
	}

	virtual size_t classesCount() const override
	{
		return 1;
	}

	virtual size_t classForIndex(size_t index) override
	{
		return _classNum;
	}

	virtual std::string modelStringForClass(size_t classNum)
	{
		return modelString;
	}

	torch::data::Example<torch::Tensor, torch::Tensor> get(size_t index) override
	{
		assert(index < examples.size());

		return examples[index];
	}

	c10::optional<size_t> size() const override
	{
		return examples.size();
	}
};
