#pragma once
#include <filesystem>
#include <kisstype/spectra.h>
#include <memory>

#include "microtar.h"
#include "data/loaders/eisspectradataset.h"

class TarDataset: public EisSpectraDataset
{
private:
	std::filesystem::path path;
	mtar_t tar;
	bool open = false;
	std::vector<std::string> labels;

protected:

	struct File
	{
		std::string path;
		size_t pos;
		size_t size;
	};
	std::shared_ptr<std::vector<File>> files;

	void loadTar(const std::filesystem::path& path);
	eis::Spectra loadSpectraHeaderAtCurrentPos(size_t size);
	eis::Spectra loadSpectraAtCurrentPos(size_t size);
	virtual eis::Spectra loadSpectraHeaderAtIndex(size_t index) override;
	virtual eis::Spectra loadSpectraAtIndex(size_t index) override;

public:
	TarDataset() = default;
	TarDataset(const TarDataset& in);
	virtual ~TarDataset();

	virtual TarDataset& operator=(const TarDataset& in);
};
