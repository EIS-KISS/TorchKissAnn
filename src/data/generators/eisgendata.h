#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <optional>
#include <eisgenerator/model.h>
#include <string>
#include <data/eisdataset.h>

class EisGeneratorDataset :
public EisDataset<EisGeneratorDataset>
{
public:
	static constexpr bool PRINT = false;
	static constexpr size_t DEFAULT_EXAMPLE_COUNT = 1e8;

public:
	std::vector<eis::Model> models;
	std::vector< std::vector<eis::Range> > modelRanges;
	std::vector<size_t> modelSweepSizes;
	eis::Range omega;
	double noise;


public:
	explicit EisGeneratorDataset(int64_t outputSize = 100, double noiseI = 0);
	EisGeneratorDataset(const EisGeneratorDataset& in) = default;

	virtual torch::data::Example<torch::Tensor, torch::Tensor> get(size_t index) override;

	virtual torch::Tensor classWeights() override;

	inline void setOmegaRange(eis::Range range){omega = range;}

	void addModel(const eis::Model& model, std::vector<eis::Range> modelRange);

	EisGeneratorDataset* getTestDataset();

	virtual size_t classesCount() const override;
	size_t frequencies();

	virtual size_t classForIndex(size_t index) override;

	virtual std::string modelStringForClass(size_t classNum) override;

	static EisGeneratorDataset* getNeisGeneratedDataset(int64_t outputSize = 100, size_t desiredCount = DEFAULT_EXAMPLE_COUNT,
														double noise = 0, bool inductivity = false, bool log = true);

	virtual c10::optional<size_t> size() const override;
};
