//
// TorchKissAnn - A collection of tools to train various types of Machine learning
// algorithms on various types of EIS data
// Copyright (C) 2025 Carl Klemm <carl@uvos.xyz>
//
// This file is part of TorchKissAnn.
//
// TorchKissAnn is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// TorchKissAnn is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with TorchKissAnn.  If not, see <http://www.gnu.org/licenses/>.
//

#include "dirdataset.h"
#include "data/eisdataset.h"
#include <vector>

#include "indicators.hpp"

void DirDataset::loadDir(const std::filesystem::path& path)
{
	files.reset(new std::vector<File>);

	const std::filesystem::path directoryPath{path};
	if(!std::filesystem::is_directory(directoryPath))
		throw dataset_error(directoryPath.string() + " is not a valid directory");

	indicators::BlockProgressBar bar(
		indicators::option::BarWidth(50),
		indicators::option::PrefixText("Loading " + path.string() + ": "),
		indicators::option::ShowElapsedTime(true),
		indicators::option::MaxProgress(files->size())
	);

	for(const std::filesystem::directory_entry& dirent : std::filesystem::directory_iterator{directoryPath})
	{
		bar.tick();
		if(!dirent.is_regular_file() || dirent.path().extension() != ".csv")
			continue;
		Log(Log::DEBUG)<<"Using: "<<dirent.path().filename();
		files->push_back({dirent.path()});
	}
	if(files->size() < 20)
		Log(Log::WARN)<<"found few valid files in "<<directoryPath;
}

eis::Spectra DirDataset::loadSpectraHeaderAtIndex(size_t index)
{
	if(!files || index >= files->size())
		throw dataset_error("index " + std::to_string(index) + " is out of range for dataset");
	return eis::Spectra::loadHeaderFromDisk((*files)[index].path);
}

eis::Spectra DirDataset::loadSpectraAtIndex(size_t index)
{
	if(!files ||index >= files->size())
		throw dataset_error("index " + std::to_string(index) + " is out of range for dataset");
	return eis::Spectra::loadFromDisk((*files)[index].path);
}
