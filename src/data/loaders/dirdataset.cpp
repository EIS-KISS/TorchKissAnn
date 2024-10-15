#include "dirdataset.h"
#include "data/eisdataset.h"

void DirDataset::loadDir(const std::filesystem::path& path)
{
	const std::filesystem::path directoryPath{path};
	if(!std::filesystem::is_directory(directoryPath))
		throw dataset_error(directoryPath.string() + " is not a valid directory");

	for(const std::filesystem::directory_entry& dirent : std::filesystem::directory_iterator{directoryPath})
	{
		if(!dirent.is_regular_file() || dirent.path().extension() != ".csv")
			continue;
		Log(Log::DEBUG)<<"Using: "<<dirent.path().filename();
		files.push_back({dirent.path()});
	}
	if(files.size() < 20)
		Log(Log::WARN)<<"found few valid files in "<<directoryPath;
}

eis::Spectra DirDataset::loadSpectraHeaderAtIndex(size_t index)
{
	if(index >= files.size())
		throw dataset_error("index " + std::to_string(index) + " is out of range for dataset");
	return eis::Spectra::loadHeaderFromDisk(files[index].path);
}

eis::Spectra DirDataset::loadSpectraAtIndex(size_t index)
{
	if(index >= files.size())
		throw dataset_error("index " + std::to_string(index) + " is out of range for dataset");
	return eis::Spectra::loadFromDisk(files[index].path);
}
