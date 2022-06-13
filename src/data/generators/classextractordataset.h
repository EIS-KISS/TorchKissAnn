#pragma once
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <optional>

template <typename DatasetSelf>
class ClassExtractorDataset :
public torch::data::datasets::Dataset<ClassExtractorDataset<DatasetSelf>, torch::data::Example<torch::Tensor, torch::Tensor>>
{
private:
	torch::data::datasets::Dataset<DatasetSelf, torch::data::Example<torch::Tensor, torch::Tensor>>* _dataset;

	std::vector<size_t> indexList;

public:

	ClassExtractorDataset(torch::data::datasets::Dataset<DatasetSelf, torch::data::Example<torch::Tensor, torch::Tensor>>* dataset, size_t classNum):
	_dataset(dataset)
	{
		size_t dataSetSize = _dataset->size().value();
		for(size_t i = 0; i < dataSetSize; ++i)
		{
			torch::data::Example<torch::Tensor, torch::Tensor> example = _dataset->get(i);
			if(example.target.item<int64_t>() == classNum)
			{
				indexList.push_back(i);
			}
		}
	}

	torch::data::Example<torch::Tensor, torch::Tensor> get(size_t index) override
	{
		assert(index < indexList.size());

		return _dataset->get(indexList[index]);
	}

	c10::optional<size_t> size() const override
	{
		return indexList.size();
	}
};
