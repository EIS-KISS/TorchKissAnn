#include "tardataset.h"
#include "data/eisdataset.h"
#include "indicators.hpp"
#include "microtar.h"
#include <filesystem>

TarDataset::~TarDataset()
{
	if(open)
		mtar_close(&tar);
}

TarDataset::TarDataset(const TarDataset& in)
{
	operator=(in);
}

TarDataset& TarDataset::operator=(const TarDataset& in)
{
	files = in.files;
	path = in.path;
	if(open)
		mtar_close(&tar);
	if(in.open)
	{
		int ret = mtar_open(&tar, path.c_str(), "r");
		if(ret != 0)
			throw dataset_error("Unable to reopen tar file at " + path.string());
		open = true;
	}

	return *this;
}

void TarDataset::loadTar(const std::filesystem::path& path)
{
	files.reset(new std::vector<File>);

	int ret = mtar_open(&tar, path.c_str(), "r");
	if(ret != 0)
		throw dataset_error(path.string() + " is not a valid tar archive");

	this->path = path;
	open = true;

	mtar_header_t header;

	indicators::BlockProgressBar bar(
		indicators::option::BarWidth(50),
		indicators::option::PrefixText("Loading " + path.string() + ": "),
		indicators::option::ShowElapsedTime(true),
		indicators::option::ShowRemainingTime(true)
	);

	indicators::show_console_cursor(false);

	double tarSize = static_cast<double>(std::filesystem::file_size(path));
	size_t progress = 0;

	while((mtar_read_header(&tar, &header)) != MTAR_ENULLRECORD)
	{
		if(static_cast<size_t>((tar.pos/tarSize)*100) > progress)
		{
			progress = static_cast<size_t>((tar.pos/tarSize)*100);
			bar.set_progress(progress);
		}
		if(header.type == MTAR_TREG)
			files->push_back({.path =  header.name, .pos = tar.pos, .size = header.size});
		mtar_next(&tar);
	}

	bar.mark_as_completed();
	indicators::show_console_cursor(true);
}

eis::Spectra TarDataset::loadSpectraHeaderAtCurrentPos(size_t size)
{
	char* filebuffer = new char[size+1];
	filebuffer[size] = '\0';
	int ret = mtar_read_data(&tar, filebuffer, size);
	if(ret != 0)
	{
		Log(Log::ERROR)<<"Unable to read from tar archive";
		assert(ret == 0);
	}
	std::stringstream ss(filebuffer);

	eis::Spectra spectra = eis::Spectra::loadHeaderFromStream(ss);
	delete[] filebuffer;

	return spectra;
}

eis::Spectra TarDataset::loadSpectraHeaderAtIndex(size_t index)
{
	if(!files || index >= files->size())
		throw dataset_error("index " + std::to_string(index) + " is out of range for dataset");
	mtar_seek(&tar, (*files)[index].pos);
	return loadSpectraHeaderAtCurrentPos((*files)[index].size);
}

eis::Spectra TarDataset::loadSpectraAtCurrentPos(size_t size)
{
	char* filebuffer = new char[size+1];
	filebuffer[size] = '\0';
	int ret = mtar_read_data(&tar, filebuffer, size);
	if(ret != 0)
	{
		Log(Log::ERROR)<<"Unable to read from tar archive";
		assert(ret == 0);
	}
	std::stringstream ss(filebuffer);

	eis::Spectra spectra = eis::Spectra::loadFromStream(ss);
	delete[] filebuffer;

	if(labels.empty())
		labels = spectra.labelNames;
	else if(!std::equal(labels.begin(), labels.end(), spectra.labelNames.begin(), spectra.labelNames.end()))
		throw dataset_error(std::string("Not all spectra in ") + path.string() + " have the same labels");

	return spectra;
}

eis::Spectra TarDataset::loadSpectraAtIndex(size_t index)
{
	if(!files || index >= files->size())
		throw dataset_error("index " + std::to_string(index) + " is out of range for dataset");
	mtar_seek(&tar, (*files)[index].pos);
	return loadSpectraAtCurrentPos((*files)[index].size);
}
