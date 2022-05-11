#pragma once

#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>

#include <cstddef>
#include <vector>
#include <optional>

#include <eisgenerator/model.h>

class EisGeneratorDataset :
public torch::data::datasets::Dataset<EisGeneratorDataset, torch::data::Example<torch::Tensor, torch::Tensor>>
{
public:
	static constexpr int OUTPUT_SIZE = 50;
	static constexpr int SWEEP_DEPTH = 100;
	static constexpr bool PRINT = false;
private:
	std::vector<eis::Model> models;
	std::vector< std::vector<eis::Range> > modelRanges;
	std::vector<size_t> modelSweepSizes;
	eis::Range omega;

public:
	explicit EisGeneratorDataset();

	torch::data::Example<torch::Tensor, torch::Tensor> get(size_t index) override;

	size_t modelCount();
	size_t frequencies();

	c10::optional<size_t> size() const override;
};
