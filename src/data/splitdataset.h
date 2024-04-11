#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "eisdataset.h"
#include "randomgen.h"

template <typename SourceDataset>
class SplitDataset:
public EisDataset<SplitDataset<SourceDataset>>
{
	SourceDataset* dataset_;
	std::vector<size_t> indices_;

	virtual torch::data::Example<torch::Tensor, torch::Tensor> getImpl(size_t index) override
	{
		return dataset_->get(indices_.at(index));
	}

public:
	SplitDataset(SourceDataset *dataset, const std::vector<size_t>& indices):
	dataset_(dataset),
	indices_(indices)
	{}

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
		return dataset_->classForIndex(indices_.at(index));
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

template <typename SourceDataset>
std::pair<SplitDataset<SourceDataset>, SplitDataset<SourceDataset>> createSplitDataset(SourceDataset* dataset, double split)
{
	size_t size = dataset->size().value();

	std::vector<size_t> aIdx(size);
	std::vector<size_t> bIdx;
	bIdx.reserve(size*split);

	for(size_t i = 0; i < aIdx.size(); ++i)
		aIdx[i] = i;

	while(aIdx.size() > size*(1-split))
	{
		size_t index = rd::rand(aIdx.size()-1);
		bIdx.push_back(aIdx[index]);
		aIdx.erase(aIdx.begin()+index);
	}
	return std::pair<SplitDataset<SourceDataset>, SplitDataset<SourceDataset>>(SplitDataset<SourceDataset>(dataset, aIdx), SplitDataset<SourceDataset>(dataset, bIdx));
}
