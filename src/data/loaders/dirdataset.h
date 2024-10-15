#pragma once
#include <filesystem>
#include <kisstype/spectra.h>
#include <eisgenerator/translators.h>

#include "data/loaders/eisspectradataset.h"

class DirDataset: public EisSpectraDataset
{
protected:
	struct File
	{
		std::filesystem::path path;
	};

	std::vector<File> files;

	void loadDir(const std::filesystem::path& path);
	virtual eis::Spectra loadSpectraAtIndex(size_t index) override;
	virtual eis::Spectra loadSpectraHeaderAtIndex(size_t index) override;

public:
	DirDataset() = default;
};
