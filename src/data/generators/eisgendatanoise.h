#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <eisgenerator/model.h>
#include <string>
#include <data/eisdataset.h>
#include <filesystem>
#include <memory>
#include <eisnoise/eisnoise.h>


class EisGeneratorDatasetNoise :
public EisDataset<EisGeneratorDatasetNoise>
{
	struct ModelData
	{
		std::shared_ptr<eis::Model> model;
		std::vector<size_t> indecies;
		size_t totalCount;
		size_t classNum;
	};

public:
	static constexpr bool PRINT = false;
	static constexpr size_t DEFAULT_EXAMPLE_COUNT = 1e8;

private:
	std::vector<ModelData> models;

	eis::Range omega;
	EisNoise noise;
	size_t classCounter = 0;

private:
	std::pair<size_t, size_t> getModelAndOffsetForIndex(size_t index) const;
	void addVectorOfModels(const std::vector<std::string>& modelStrs, int64_t desiredSize, bool inductivity);

	virtual torch::data::Example<torch::Tensor, torch::Tensor> getImpl(size_t index) override;
	ModelData* findSameClass(std::string modelStr);

public:
	explicit EisGeneratorDatasetNoise(int64_t outputSize = 100);
	explicit EisGeneratorDatasetNoise(const std::filesystem::path& path, int64_t desiredSize, int64_t outputSize, bool inductivity);
	explicit EisGeneratorDatasetNoise(std::istream& is, int64_t outputSize, int64_t desiredSize, bool inductivity);
	explicit EisGeneratorDatasetNoise(const char* cStr, size_t cStrLen, int64_t desiredSize, int64_t outputSize, bool inductivity);

	EisGeneratorDatasetNoise(const EisGeneratorDatasetNoise& in) = default;
	void addModel(const eis::Model& model, size_t targetExamples);
	void addModel(std::shared_ptr<eis::Model> model, size_t targetExamples);
	EisGeneratorDatasetNoise* getTestDataset();
	size_t frequencies();
	void setOmegaRange(eis::Range range);

	virtual torch::Tensor classWeights() override;
	virtual torch::Tensor classCounts() override;
	virtual size_t classesCount() const override;
	virtual size_t classForIndex(size_t index) override;
	virtual std::string modelStringForClass(size_t classNum) override;
	virtual c10::optional<size_t> size() const override;

	static EisGeneratorDatasetNoise* newDefaultDataset(int64_t outputSize, int64_t desiredSize, bool inductivity = false);
};
