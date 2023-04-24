#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <filesystem>

#include "globals.h"
#include "data/eisdataset.h"

class EisDirDataset : public EisDataset<EisDirDataset>
{
private:

	struct FileNameStr
	{
		std::filesystem::path path;
		size_t classNum;
	};

	std::vector<EisDirDataset::FileNameStr> fileNames;
	FileType fileType;
	size_t outputSize;
	std::vector<std::string> modelStrs;

	virtual torch::data::Example<torch::Tensor, torch::Tensor> getImpl(size_t index) override;

public:
	explicit EisDirDataset(const std::string& dirName, int64_t outputSizeI = 100);
	EisDirDataset(const EisDirDataset& in) = default;

	torch::Tensor modelWeights();

	virtual c10::optional<size_t> size() const override;

	virtual size_t classesCount() const;
	virtual size_t classForIndex(size_t index);
	virtual std::string modelStringForClass(size_t classNum) override;
	virtual torch::Tensor classCounts() override;
};
