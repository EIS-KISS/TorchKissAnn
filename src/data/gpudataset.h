#include <stdint.h>

#include "eisdataset.h"
#include "globals.h"

template <typename SourceDataset>
class GpuDataset:
public EisDataset<GpuDataset<SourceDataset>>
{
	SourceDataset *dataset_;
	torch::Tensor data;
	torch::Tensor tagets;

	virtual torch::data::Example<torch::Tensor, torch::Tensor> getImpl(size_t index) override
	{
		return torch::data::Example<torch::Tensor, torch::Tensor>(data.select(0, index), tagets.select(0, index));
	}

public:
	GpuDataset(SourceDataset *dataset): dataset_(dataset)
	{
		int64_t size = dataset->size().value();

		torch::data::Example<torch::Tensor, torch::Tensor> referanceExample = dataset->get(0);

		torch::TensorOptions options;
		options = options.layout(torch::kStrided);
		options = options.device(torch::kCPU);
		options = options.dtype(torch::kFloat32);
		data = torch::empty({referanceExample.data.numel(), size}, options);
		tagets = torch::empty({referanceExample.target.numel(), size}, options);

		torch::data::DataLoaderOptions dataLoaderOptions;
		dataLoaderOptions = dataLoaderOptions.batch_size(size).workers(JOBS);
		auto DataLoader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(dataset->map(torch::data::transforms::Stack<>()), options);

	}

	virtual size_t classesCount() const override
	{
		return dataset_->classesCount();
	}

	virtual c10::optional<size_t> size() const override
	{
		return dataset_->size();
	}

	virtual size_t classForIndex(size_t index)
	{
		return dataset_->classForIndex(index);
	}

	virtual std::string modelStringForClass(size_t classNum) override
	{
		return dataset_->modelStringForClass(classNum);
	}

	virtual bool isMulticlass() override
	{
		return dataset_->isMulticlass();
	}

	virtual torch::Tensor classWeights() override
	{
		return dataset_->classWeights();
	}
};
