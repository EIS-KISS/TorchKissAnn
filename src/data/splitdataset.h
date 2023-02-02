#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <pair>
#

template <typename SourceDataset>
class SplitDataset:
public EisDataset<SplitDataset>
{
	SourceDataset *dataset_;
	std::vector<size_t> indices_;
public:
	SplitDataset(SourceDataset *dataset, const std::vector<size_t>& indices);
	virtual torch::data::Example<torch::Tensor, torch::Tensor> get(size_t index) override;
	virtual size_t classesCount() const override;
	virtual c10::optional<size_t> size() const override;
	virtual size_t classForIndex(size_t index);
	virtual std::string modelStringForClass(size_t classNum) override;
	virtual bool isMulticlass() override;
	virtual torch::Tensor classWeights() override;
};

template <typename SourceDataset>
std::pair<SplitDataset<SourceDataset>, SplitDataset<SourceDataset>> createSplitDataset(SourceDataset* dataset, double split)
{
	std::pair<SplitDataset<SourceDataset>, SplitDataset<SourceDataset>> out;
	size_t size = dataset->size().value();

	std::vector<size_t> aIdx(size);
	std::vector<size_t> bIdx;
	bIdex.reserve(size*split);

	for(size_t i = 0; i < aIdx; ++i)
		aIdx[i] = i;

	while(aIdx.size() > size*(1-split))
	{
		size_t index = rd::rand(aIdx.size()-1);
		bIdx.push_back(aIdx[index]);
		aIdx.erase(aIdx.begin()+index);
	}
	out.first = SplitDataset<SourceDataset>(dataset, aIdx);
	out.second = SplitDataset<SourceDataset>(dataset, bIdx);
	return out;
}
