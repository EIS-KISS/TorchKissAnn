#pragma once
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <optional>
#include <limits>

#include "constants.h"

template <typename DatasetSelf>
class DeduplatorDataset :
public torch::data::datasets::Dataset<DeduplatorDataset<DatasetSelf>, torch::data::Example<torch::Tensor, torch::Tensor>>
{
	static constexpr size_t BATCH_SIZE = 1000;
	static constexpr float epsilon = 0.05;
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
		options = options.dtype(torch::kInt8);
		options = options.layout(torch::kStrided);
		options = options.device(torch::kCPU);

		for(size_t i = 0; i < dataSetSize; ++i)
		{
			auto masterExample = _dataset->get(i);

			targets[i] = torch::zeros({classCount}, options);
			targets[i][masterExample.target.template item<int8_t>()] = 1;

			//std::cout<<"targets:\n"<<targets[i].sizes()<<std::endl;

			float smalest = std::numeric_limits<float>::max();
			for(auto& batch : *dataLoader)
			{
				torch::Tensor master = masterExample.data.repeat({batch.data.size(0), 1});
				master = (master-batch.data).pow(2).sum(1, true).div(masterExample.data.size(0));
				//std::cout<<"master proc:\n"<<master<<std::endl;
				float* masterDataPtr = master.contiguous().template data_ptr<float>();
				for(size_t j = 0; j < static_cast<size_t>(master.size(0)); ++j)
				{
					if(batch.target[j].template item<int8_t>() != masterExample.target.template item<int8_t>())
					{
						if(masterDataPtr[j] < smalest)
							smalest = masterDataPtr[j];
						if(masterDataPtr[j] < epsilon)
						{
							targets[i][batch.target[j].template item<int8_t>()] = 1;
						}
					}
				}
			}
			Log(Log::DEBUG)<<"Taret for "<<i<<":\n"<<targets[i]<<'\n'<<"smalest: "<<smalest;
			Log(Log::INFO)<<i<<'\\'<<dataSetSize;
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
