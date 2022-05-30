#pragma once

#include <cstdint>
#include <sys/types.h>
#include <random>

class TrashDataset :
public torch::data::datasets::Dataset<TrashDataset, torch::data::Example<torch::Tensor, torch::Tensor>>
{
	int64_t outputSize;
	int64_t classCount;
public:
	explicit TrashDataset(int64_t outputSizeIn = 100, int64_t classCountIn = 6);

	torch::data::Example<torch::Tensor, torch::Tensor> get(size_t index) override;

	c10::optional<size_t> size() const override;
};
