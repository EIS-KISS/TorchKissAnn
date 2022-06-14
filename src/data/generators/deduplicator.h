#pragma once
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <optional>

#include "constants.h"

template <typename DatasetSelf>
class DeduplatorDataset :
public torch::data::datasets::Dataset<ClassExtractorDataset<DatasetSelf>, torch::data::Example<torch::Tensor, torch::Tensor>>
{
	static constexpr size_t BATCH_SIZE = 10000;
	static constexpr float epsilon = 0.01;
private:
	torch::data::datasets::Dataset<DatasetSelf, torch::data::Example<torch::Tensor, torch::Tensor>>* _dataset;
	std::vector<torch::Tensor> targets;

public:

	DeduplatorDataset(torch::data::datasets::Dataset<DatasetSelf, torch::data::Example<torch::Tensor, torch::Tensor>>* dataset, size_t classCount):
	_dataset(dataset)
	{
		size_t dataSetSize = dataset->size().value();
		targets.resize(dataSetSize);

		torch::data::DataLoaderOptions loaderOptions;
		loaderOptions = loaderOptions.batch_size(BATCH_SIZE);
		loaderOptions = loaderOptions.workers(JOBS);
		loaderOptions = loaderOptions.max_jobs(JOBS*2);
		auto dataLoader = torch::data::make_data_loader(dataset->map(torch::data::transforms::Stack<>()), loaderOptions);

		torch::TensorOptions options;
		options = options.dtype(torch::kFloat32);
		options = options.layout(torch::kStrided);
		options = options.device(torch::kCPU);

		for(size_t i = 0; i < dataSetSize; ++i)
		{
			auto masterExample = _dataset->get(i);

			targets[i] = torch::zeros({classCount}, options);
			targets[i][masterExample.target.template item<int64_t>()] = 1;

			for(auto& batch : *dataLoader)
			{
				torch::Tensor master = masterExample.data.repeat({batch.data.size(0), 1});
				master = (master-batch.data).pow(2).sum(1, true).div(masterExample.data.size(1));
				float* masterDataPtr = master.contiguous().template data_ptr<float>();
				for(size_t j = 0; j < masterExample.data.size(1); ++j)
				{
					if(masterDataPtr[j] < epsilon)
					{
						targets[i][batch.target[j].template item<int64_t>()] = 1;
					}
				}
			}
		}
	}

	torch::data::Example<torch::Tensor, torch::Tensor> get(size_t index) override
	{
		assert(index < targets.size());

		torch::data::Example<torch::Tensor, torch::Tensor> example = _dataset->get(index);

		return torch::data::Example<torch::Tensor, torch::Tensor>(example.data, targets[index]);
	}

	c10::optional<size_t> size() const override
	{
		return targets.size();
	}
};
