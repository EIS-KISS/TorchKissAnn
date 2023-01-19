#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <optional>
#include <eisgenerator/model.h>
#include <string>
#include <data/eisdataset.h>
#include <set>
#include <filesystem>

class EisGeneratorDataset :
public EisDataset<EisGeneratorDataset>
{
public:
	static constexpr bool PRINT = false;
	static constexpr size_t DEFAULT_EXAMPLE_COUNT = 1e8;

private:
	std::vector<eis::Model> models;
	std::vector<size_t> modelClasses;
	std::set<std::string> modelStrings;
	std::vector<size_t> modelSweepSizes;
	eis::Range omega;
	double noise;

private:
	std::pair<eis::Model*, size_t> getModelAndOffsetForIndex(size_t index);

public:
	explicit EisGeneratorDataset(int64_t outputSize = 100, double noiseI = 0);
	EisGeneratorDataset(const EisGeneratorDataset& in) = default;

	virtual torch::data::Example<torch::Tensor, torch::Tensor> get(size_t index) override;

	virtual torch::Tensor classWeights() override;

	inline void setOmegaRange(eis::Range range){omega = range;}

	void addModel(const eis::Model& model);

	EisGeneratorDataset* getTestDataset();

	virtual size_t classesCount() const override;
	size_t frequencies();

	virtual size_t classForIndex(size_t index) override;

	virtual std::string modelStringForClass(size_t classNum) override;

	static EisGeneratorDataset* getNeisGeneratedDataset(int64_t outputSize = 100, double noise = 0, bool inductivity = false, const std::filesystem::path* path = nullptr);

	virtual c10::optional<size_t> size() const override;
};
