#pragma once

#include <vector>
#include <string>
#include <filesystem>
#include <kisstype/type.h>

#include "data/loaders/dirdataset.h"
#include "data/eisdataset.h"

class EisDirDataset : public DirDataset, public EisDataset<EisDirDataset>
{
private:
	std::vector<std::string> modelStrs;
	std::vector<size_t> classIndexes;

	virtual torch::data::Example<torch::Tensor, torch::Tensor> getImpl(size_t index) override;

public:
	explicit EisDirDataset(const std::filesystem::path& path);
	EisDirDataset(const EisDirDataset& in) = default;

	torch::Tensor modelWeights();

	virtual c10::optional<size_t> size() const override;

	virtual size_t outputSize() const override;
	virtual std::string outputName(size_t output) override;
	virtual torch::Tensor classCounts() override;
	virtual std::vector<std::pair<std::string, int64_t>> extraInputs() override;
	virtual c10::optional<torch::Tensor> frequencies() override;
};
