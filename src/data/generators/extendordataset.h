#pragma once
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <optional>
#include <mutex>
#include <memory>

#include "data/eisdataset.h"
#include "randomgen.h"

template <typename ParentType>
class  ExtendorDataset : public EisDataset<ExtendorDataset<ParentType>>
{
private:
	EisDataset<ParentType>* _dataset;
	std::shared_ptr<std::mutex> datasetMutex;
	size_t _factor;


	virtual torch::data::Example<torch::Tensor, torch::Tensor> getImpl(size_t index) override
	{
		//datasetMutex->lock();
		torch::data::Example<torch::Tensor, torch::Tensor> example = _dataset->get(index % _dataset->size().value());
		//datasetMutex->unlock();

		if(index > _dataset->size().value())
		{
			torch::TensorOptions options;
			options = options.dtype(torch::kFloat32);
			options = options.layout(torch::kStrided);
			options = options.device(torch::kCPU);
			torch::Tensor randomNumbers = torch::randn(example.data.sizes(), options);
			randomNumbers = randomNumbers*0.001f;
			example.data = example.data + randomNumbers;
		}
		return example;
	}

public:

	ExtendorDataset(EisDataset<ParentType>* dataset, size_t factor):
	_dataset(dataset), datasetMutex(new std::mutex), _factor(factor)
	{
	}

	virtual size_t classesCount() const override
	{
		return _dataset->classesCount();
	}

	virtual size_t classForIndex(size_t index) override
	{
		return _dataset->classForIndex(index % _dataset->size().value());
	}

	virtual std::string modelStringForClass(size_t classNum)
	{
		return _dataset->modelStringForClass(classNum);
	}

	virtual c10::optional<size_t> size() const override
	{
		return _dataset->size().value()*_factor;
	}

	virtual bool isMulticlass()
	{
		return _dataset->isMulticlass();
	}

	virtual torch::Tensor classCounts()
	{
		return _dataset->classCounts();
	}

	virtual torch::Tensor classWeights()
	{
		return _dataset->classWeights();
	}
};
