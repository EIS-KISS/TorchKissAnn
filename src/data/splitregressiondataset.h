#pragma once
#include <cstdint>
#include <string>
#include <utility>

#include "data/regressiondataset.h"
#include "regressiondataset.h"
#include "randomgen.h"

template <typename SourceDataset>
class SplitRegressionDataset:
public RegressionDataset<SplitRegressionDataset<SourceDataset>>
{
	SourceDataset dataset_;
	std::vector<size_t> indices_;

	virtual torch::data::Example<torch::Tensor, torch::Tensor> getImpl(size_t index) override
	{
		return dataset_.get(indices_.at(index));
	}

public:
	SplitRegressionDataset(SourceDataset dataset, const std::vector<size_t>& indices):
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
	virtual std::pair<torch::Tensor, torch::Tensor> getTargetScalesAndBias() override;

	static std::pair<SplitRegressionDataset<SourceDataset>, SplitRegressionDataset<SourceDataset>> createSplitRegressionDataset(SourceDataset* dataset, double split);
};

template <typename SourceDataset>
size_t SplitRegressionDataset<SourceDataset>::outputSize() const
{
	return dataset_.outputSize();
}

template <typename SourceDataset>
c10::optional<size_t> SplitRegressionDataset<SourceDataset>::size() const
{
	return indices_.size();
}

template <typename SourceDataset>
size_t SplitRegressionDataset<SourceDataset>::classForIndex(size_t index)
{
	return dataset_.classForIndex(indices_.at(index));
}

template <typename SourceDataset>
std::string SplitRegressionDataset<SourceDataset>::modelStringForClass(size_t classNum)
{
	return dataset_.modelStringForClass(classNum);
}

template <typename SourceDataset>
bool SplitRegressionDataset<SourceDataset>::isMulticlass()
{
	return dataset_.isMulticlass();
}

template <typename SourceDataset>
torch::Tensor SplitRegressionDataset<SourceDataset>::classCounts()
{
	return dataset_.classCounts();
}

template <typename SourceDataset>
torch::Tensor SplitRegressionDataset<SourceDataset>::classWeights()
{
	return dataset_.classWeights();
}

template <typename SourceDataset>
size_t SplitRegressionDataset<SourceDataset>::inputSize()
{
	return dataset_.inputSize();
}

template <typename SourceDataset>
c10::optional<torch::Tensor> SplitRegressionDataset<SourceDataset>::freqRange()
{
	return dataset_.freqRange();
}

template <typename SourceDataset>
std::string  SplitRegressionDataset<SourceDataset>::dataLabel() const
{
	return dataset_.dataLabel();
}

template <typename SourceDataset>
std::vector<std::pair<std::string, int64_t>> SplitRegressionDataset<SourceDataset>::extraInputs()
{
	return dataset_.extraInputs();
}

template <typename SourceDataset>
std::pair<torch::Tensor, torch::Tensor> SplitRegressionDataset<SourceDataset>::getTargetScalesAndBias()
{
	return dataset_.getTargetScalesAndBias();
}

template <typename SourceDataset>
std::pair<SplitRegressionDataset<SourceDataset>, SplitRegressionDataset<SourceDataset>> SplitRegressionDataset<SourceDataset>::createSplitRegressionDataset(SourceDataset* dataset, double split)
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
	return std::pair<SplitRegressionDataset<SourceDataset>, SplitRegressionDataset<SourceDataset>>(SplitRegressionDataset<SourceDataset>(*dataset, aIdx), SplitRegressionDataset<SourceDataset>(*dataset, bIdx));
}
