#pragma once

#include <eisgenerator/eistype.h>
#include <eisgenerator/model.h>
#include <string>
#include <filesystem>

#include "data/regressiondataset.h"
#include "microtar.h"

class ParameterRegressionDatasetLoader :
public RegressionDataset<ParameterRegressionDatasetLoader>
{
private:
	virtual torch::data::Example<torch::Tensor, torch::Tensor> getImpl(size_t index) override;

	size_t parameterCount;
	std::vector<std::pair<std::filesystem::path, size_t>> files;
	std::filesystem::path path;
	bool drt;
	mtar_t tar;
	std::vector<int64_t> dataSizes = {};

	eis::EisSpectra loadSpectra(size_t index);


public:
	explicit ParameterRegressionDatasetLoader(const std::filesystem::path& pathI, bool drtI = true);
	ParameterRegressionDatasetLoader(const ParameterRegressionDatasetLoader& in);
	ParameterRegressionDatasetLoader& operator=(const ParameterRegressionDatasetLoader& in);
	~ParameterRegressionDatasetLoader();

	virtual size_t outputSize() const override;
	virtual size_t classForIndex(size_t index) override;
	virtual std::string modelStringForClass(size_t classNum) override;
	virtual c10::optional<size_t> size() const override;
	virtual bool isMulticlass() override;
	virtual std::string dataLabel() const override;
	virtual c10::optional<torch::Tensor> freqRange() override;
	virtual std::pair<torch::Tensor, torch::Tensor> getTargetScalesAndBias() override;
};
