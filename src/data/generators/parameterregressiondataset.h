#pragma once

#include <eisgenerator/model.h>
#include <string>

#include "data/eisdataset.h"

class ParameterRegressionDatasetGenerator :
public EisDataset<ParameterRegressionDatasetGenerator>
{
public:
	static constexpr size_t DEFAULT_EXAMPLE_COUNT = 1e8;

private:
	eis::Model model;

	eis::Range omega;
	double noise;
	size_t sweepCount;
	size_t parameterCount;
	bool drt;

private:
	virtual torch::data::Example<torch::Tensor, torch::Tensor> getImpl(size_t index) override;

public:
	explicit ParameterRegressionDatasetGenerator(const std::string& model, int64_t outputSize = 100, double noiseI = 0, bool drtI = true, size_t targetSize = 0);
	ParameterRegressionDatasetGenerator(const ParameterRegressionDatasetGenerator& in) = default;

	void setOmegaRange(eis::Range range);

	virtual size_t classesCount() const override;
	virtual size_t classForIndex(size_t index) override;
	virtual std::string modelStringForClass(size_t classNum) override;
	virtual c10::optional<size_t> size() const override;
	virtual bool isMulticlass() override;
	virtual std::string dataLabel() const override;
	virtual c10::optional<torch::Tensor> freqRange() override;
};
