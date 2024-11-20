#pragma once

#include <string>
#include <filesystem>
#include <kisstype/spectra.h>

#include "data/regressiondataset.h"
#include "data/loaders/tardataset.h"

class RegressionLoaderTar :
public TarDataset, public RegressionDataset<RegressionLoaderTar>
{
protected:
	virtual torch::data::Example<torch::Tensor, torch::Tensor> getImpl(size_t index) override;

	size_t outputCount;

public:
	explicit RegressionLoaderTar(const std::filesystem::path& pathI);

	virtual size_t outputSize() const override;
	virtual std::string outputName(size_t output) override;
	virtual c10::optional<size_t> size() const override;
	virtual bool isMulticlass() override;
	virtual std::vector<std::pair<std::string, int64_t>> extraInputs() override;
	virtual const std::string targetName() override;
	virtual c10::optional<torch::Tensor> frequencies() override;
};
