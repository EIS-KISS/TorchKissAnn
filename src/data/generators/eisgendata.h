#pragma once

#include <cstdint>

#include <cstddef>
#include <vector>
#include <optional>
#include <eisgenerator/model.h>
#include <string>

class EisGeneratorDataset :
public torch::data::datasets::Dataset<EisGeneratorDataset, torch::data::Example<torch::Tensor, torch::Tensor>>
{
public:
	static constexpr bool PRINT = false;
	static constexpr int DEFAULT_SWEEP_DEPTH = 20;

private:
	std::vector<eis::Model> models;
	std::vector< std::vector<eis::Range> > modelRanges;
	std::vector<size_t> modelSweepSizes;
	eis::Range omega;
	double noise;


public:
	explicit EisGeneratorDataset(int64_t outputSize = 100, double noiseI = 0);
	EisGeneratorDataset(const EisGeneratorDataset& in) = default;

	torch::data::Example<torch::Tensor, torch::Tensor> get(size_t index) override;

	torch::Tensor modelWeights();

	inline void setOmegaRange(eis::Range range){omega = range;}

	void addModel(const eis::Model& model, std::vector<eis::Range> modelRange);

	static void filterData(std::vector<eis::DataPoint>& data, size_t outputSize);

	EisGeneratorDataset* getTestDataset();

	size_t modelCount();
	size_t frequencies();

	std::string getModelStrFromClass(size_t classNum);

	static EisGeneratorDataset* getNeisGeneratedDataset(int64_t outputSize = 100, int sweepDepth = DEFAULT_SWEEP_DEPTH, double noise = 0);
	static EisGeneratorDataset* getNeisGeneratedDatasetWithBLl(int64_t outputSize = 100, int sweepDepth = DEFAULT_SWEEP_DEPTH, double noise = 0);

	c10::optional<size_t> size() const override;
};
