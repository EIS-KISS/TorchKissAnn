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

	virtual size_t outputSize() const override;
	virtual c10::optional<size_t> size() const override;
	virtual size_t classForIndex(size_t index) override;
	virtual std::string modelStringForClass(size_t classNum) override;
	virtual bool isMulticlass() override;
	virtual torch::Tensor classCounts() override;
	virtual torch::Tensor classWeights() override;
	virtual size_t inputSize() override;
	virtual c10::optional<torch::Tensor> freqRange() override;
	virtual std::string dataLabel() const override;
	virtual std::vector<std::pair<std::string, int64_t>> extraInputs() override;

	static std::pair<SplitDataset<SourceDataset>, SplitDataset<SourceDataset>> createSplitDataset(SourceDataset* dataset, double split);
};


template <typename SourceDataset>
size_t SplitDataset<SourceDataset>::outputSize() const
{
	return dataset_->outputSize();
}

template <typename SourceDataset>
c10::optional<size_t> SplitDataset<SourceDataset>::size() const
{
	return indices_.size();
}

template <typename SourceDataset>
size_t SplitDataset<SourceDataset>::classForIndex(size_t index)
{
	return dataset_->classForIndex(indices_.at(index));
}

template <typename SourceDataset>
std::string SplitDataset<SourceDataset>::modelStringForClass(size_t classNum)
{
	return dataset_->modelStringForClass(classNum);
}

template <typename SourceDataset>
bool SplitDataset<SourceDataset>::isMulticlass()
{
	return dataset_->isMulticlass();
}

template <typename SourceDataset>
torch::Tensor SplitDataset<SourceDataset>::classCounts()
{
	return dataset_->classCounts();
}

template <typename SourceDataset>
torch::Tensor SplitDataset<SourceDataset>::classWeights()
{
	return dataset_->classWeights();
}

template <typename SourceDataset>
size_t SplitDataset<SourceDataset>::inputSize()
{
	return dataset_->inputSize();
}

template <typename SourceDataset>
c10::optional<torch::Tensor> SplitDataset<SourceDataset>::freqRange()
{
	return dataset_->freqRange();
}

template <typename SourceDataset>
std::string  SplitDataset<SourceDataset>::dataLabel() const
{
	return dataset_->dataLabel();
}

template <typename SourceDataset>
std::vector<std::pair<std::string, int64_t>> SplitDataset<SourceDataset>::extraInputs()
{
	return dataset_->extraInputs();
}

template <typename SourceDataset>
std::pair<SplitDataset<SourceDataset>, SplitDataset<SourceDataset>> SplitDataset<SourceDataset>::createSplitDataset(SourceDataset* dataset, double split)
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
