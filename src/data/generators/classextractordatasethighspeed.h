#pragma once
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <optional>
#include <mutex>

template <typename DatasetSelf>
class ClassExtractorDatasetHighSpeed :
public torch::data::datasets::Dataset<ClassExtractorDatasetHighSpeed<DatasetSelf>, torch::data::Example<torch::Tensor, torch::Tensor>>
{
private:
	std::vector<torch::data::Example<torch::Tensor, torch::Tensor>> examples;
	std::vector<size_t> indexList;

public:

	ClassExtractorDatasetHighSpeed(torch::data::datasets::Dataset<DatasetSelf, torch::data::Example<torch::Tensor, torch::Tensor>>* dataset, size_t classNum)
	{
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
