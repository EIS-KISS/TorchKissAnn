#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <filesystem>
#include <set>

#include "constants.h"
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
	std::set<std::string> modelStrs;

public:
	explicit EisDirDataset(const std::string& dirName, const std::string& prefix, FileType fileTypeI,
	                       const std::string& extension = "*", int64_t outputSizeI = 100);
	EisDirDataset(const EisDirDataset& in) = default;

	virtual torch::data::Example<torch::Tensor, torch::Tensor> get(size_t index) override;

	torch::Tensor modelWeights();

	virtual c10::optional<size_t> size() const override;

	virtual size_t classesCount() const;
	virtual size_t classForIndex(size_t index);
	virtual std::string modelStringForClass(size_t classNum) override;

	virtual torch::Tensor classWeights()
	{
		torch::TensorOptions options;
		options = options.dtype(torch::kFloat32);
		options = options.layout(torch::kStrided);
		options = options.device(torch::kCPU);
		return torch::ones({static_cast<int64_t>(classesCount())}, options);
	}
};
