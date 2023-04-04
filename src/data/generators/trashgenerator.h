#pragma once

#include <cstdint>
#include <sys/types.h>
#include <random>

#include "data/eisdataset.h"

class TrashDataset : public EisDataset<TrashDataset>
{
	int64_t outputSize;
	int64_t classCount;

	virtual torch::data::Example<torch::Tensor, torch::Tensor> getImpl(size_t index) override;

public:
	explicit TrashDataset(int64_t outputSizeIn = 100, int64_t classCountIn = 6);

	virtual c10::optional<size_t> size() const override;

	virtual size_t classesCount() const override;
	virtual size_t classForIndex(size_t index) override;
	virtual std::string modelStringForClass(size_t classNum) override;
};
