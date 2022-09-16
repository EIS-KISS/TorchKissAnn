#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>

#include "constants.h"

class FastDedupDataset :
public torch::data::datasets::Dataset<FastDedupDataset, torch::data::Example<torch::Tensor, torch::Tensor>>
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

public:
	explicit FastDedupDataset(const std::filesystem::path& fileName);
	FastDedupDataset(const FastDedupDataset& in) = default;

	torch::data::Example<torch::Tensor, torch::Tensor> get(size_t index) override;

	size_t classesCount();
	torch::Tensor modelWeights();

	std::string getModelStrFromClass(size_t classNum);

	bool isOpen() {return file.is_open();}

	c10::optional<size_t> size() const override;
};
