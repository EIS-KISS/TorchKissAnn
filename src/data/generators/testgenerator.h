#pragma once

#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>

#include <cstddef>
#include <vector>
#include <optional>

#include <eisgenerator/model.h>

class TestDataset :
public torch::data::datasets::Dataset<TestDataset, torch::data::Example<torch::Tensor, torch::Tensor>>
{
static constexpr int COUNT_CLASSES = 5;
static constexpr int OUTPUT_SIZE = 50;
public:
	explicit TestDataset();

	torch::data::Example<torch::Tensor, torch::Tensor> get(size_t index) override;

	c10::optional<size_t> size() const override;
};
