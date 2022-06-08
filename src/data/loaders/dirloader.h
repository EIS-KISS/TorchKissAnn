#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <filesystem>

#include "constants.h"

class EisDirDataset :
public torch::data::datasets::Dataset<EisDirDataset, torch::data::Example<torch::Tensor, torch::Tensor>>
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

public:
	explicit EisDirDataset(const std::string& dirName, const std::string& prefix, FileType fileTypeI, int64_t outputSizeI = 100);
	EisDirDataset(const EisDirDataset& in) = default;

	torch::data::Example<torch::Tensor, torch::Tensor> get(size_t index) override;

	size_t modelCount();
	torch::Tensor modelWeights();

	c10::optional<size_t> size() const override;
};
