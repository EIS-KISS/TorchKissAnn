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
	std::vector<std::string> modelStrings;
	std::vector<size_t> modelSweepSizes;
	eis::Range omega;
	double noise;
	size_t repetitionCount = 1;

private:
	std::pair<size_t, size_t> getModelAndOffsetForIndex(size_t index) const;
	void addVectorOfModels(const std::vector<std::string>& modelStrs, int64_t desiredSize, bool inductivity);
	size_t trueSize() const;

public:
	explicit EisGeneratorDataset(int64_t outputSize = 100, double noiseI = 0);
	explicit EisGeneratorDataset(const std::filesystem::path& path, int64_t desiredSize, int64_t outputSize, double noise, bool inductivity);
	explicit EisGeneratorDataset(std::istream& is, int64_t outputSize, int64_t desiredSize, double noise, bool inductivity);
	explicit EisGeneratorDataset(const char* cStr, size_t cStrLen, int64_t desiredSize, int64_t outputSize, double noise, bool inductivity);

	EisGeneratorDataset(const EisGeneratorDataset& in) = default;
	void addModel(const eis::Model& model);
	EisGeneratorDataset* getTestDataset();
	size_t frequencies();
	inline void setOmegaRange(eis::Range range){omega = range;}

	virtual torch::data::Example<torch::Tensor, torch::Tensor> get(size_t index) override;
	virtual torch::Tensor classWeights() override;
	virtual torch::Tensor classCounts() override;
	virtual size_t classesCount() const override;
	virtual size_t classForIndex(size_t index) override;
	virtual std::string modelStringForClass(size_t classNum) override;
	virtual c10::optional<size_t> size() const override;

	static EisGeneratorDataset* newDefaultDataset(int64_t outputSize, int64_t desiredSize, double noise = 0, bool inductivity = false);
};
