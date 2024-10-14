#pragma once

#include <string>
#include <filesystem>
#include <vector>
#include <kisstype/spectra.h>

#include "data/regressiondataset.h"
#include "microtar.h"

class RegressionLoader :
public RegressionDataset<RegressionLoader>
{
protected:
	mtar_t tar;

	struct File
	{
		std::string path;
		size_t pos;
		size_t size;
	};

	size_t outputCount;
	std::vector<File> files;
	std::filesystem::path path;
	inline static const std::string extraInputStr = "exip_";

	eis::Spectra loadSpectraAtCurrentPos(size_t size);
	std::pair<std::vector<std::string>, std::vector<std::string>> getExtraInputsAndLabelNames(const eis::Spectra& spectra);

	virtual torch::data::Example<torch::Tensor, torch::Tensor> getImpl(size_t index) override;

public:
	explicit RegressionLoader(const std::filesystem::path& pathI);
	RegressionLoader(const RegressionLoader& in);
	RegressionLoader& operator=(const RegressionLoader& in);
	~RegressionLoader();

	virtual size_t outputSize() const override;
	virtual std::string outputName(size_t output) override;
	virtual c10::optional<size_t> size() const override;
	virtual bool isMulticlass() override;
	virtual c10::optional<torch::Tensor> freqRange() override;
	virtual std::vector<std::pair<std::string, int64_t>> extraInputs() override;
};
