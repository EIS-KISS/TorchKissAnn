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
private:
	std::vector<Model> models;
	std::vector< std::vector<Model::Range> > modelRanges;
	std::vector<size_t> modelSweepSizes;
	Model::Range omega;

public:
	explicit EisGeneratorDataset();

	torch::data::Example<torch::Tensor, torch::Tensor> get(size_t index) override;

	size_t modelCount();
	size_t frequencies();

	c10::optional<size_t> size() const override;
};
