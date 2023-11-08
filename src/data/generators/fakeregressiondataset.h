#pragma once

#include <eisgenerator/model.h>
#include <string>

#include "data/eisdataset.h"

class FakeRegressionDatasetGenerator :
public EisDataset<FakeRegressionDatasetGenerator>
{
public:
	static constexpr size_t DEFAULT_EXAMPLE_COUNT = 1e8;

private:
	size_t outputSize;

private:
	virtual torch::data::Example<torch::Tensor, torch::Tensor> getImpl(size_t index) override;

public:
	explicit FakeRegressionDatasetGenerator(int64_t outputSize = 100);
	FakeRegressionDatasetGenerator(const FakeRegressionDatasetGenerator& in) = default;

	virtual size_t classesCount() const override;
	virtual size_t classForIndex(size_t index) override;
	virtual std::string modelStringForClass(size_t classNum) override;
	virtual c10::optional<size_t> size() const override;
	virtual bool isMulticlass() override;
	virtual c10::optional<torch::Tensor> freqRange() override;
};
