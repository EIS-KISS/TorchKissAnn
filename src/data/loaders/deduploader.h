#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>

#include "constants.h"
#include "data/eisdataset.h"

class FastDedupDataset : public EisDataset<FastDedupDataset>
{
	static constexpr int MIN_VERSION_MAJOR = 0;
	static constexpr int MIN_VERSION_MINOR = 1;
	static constexpr int MAX_VERSION_MAJOR = 1;

private:
	std::fstream file;
	size_t modelCount = 0;
	size_t depth = 0;
	std::vector<std::streampos> lineStarts;
	std::vector<std::string> modelStrings;
	int versionMajor = 0;
	int versionMinor = 1;

private:
	torch::data::Example<torch::Tensor, torch::Tensor> parseLine(const std::string& line);
	size_t parseOriginalClass(const std::string& line);

public:
	explicit FastDedupDataset(const std::filesystem::path& fileName);
	FastDedupDataset(const FastDedupDataset& in) = default;

	torch::data::Example<torch::Tensor, torch::Tensor> get(size_t index) override;

	virtual size_t classesCount() const override;
	virtual torch::Tensor classWeights() override;
	virtual size_t classForIndex(size_t index) override;
	virtual bool isMulticlass() {return true;}

	virtual std::string modelStringForClass(size_t classNum) override;

	bool isOpen() {return file.is_open();}

	virtual c10::optional<size_t> size() const override;
};
