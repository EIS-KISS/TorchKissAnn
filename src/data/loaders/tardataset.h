/* * TorchKissAnn - A collection of tools to train various types of Machine learning
 * algorithms on various types of EIS data
 * Copyright (C) 2025 Carl Klemm <carl@uvos.xyz>
 *
 * This file is part of TorchKissAnn.
 *
 * TorchKissAnn is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * TorchKissAnn is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with TorchKissAnn.  If not, see <http://www.gnu.org/licenses/>.
 */

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
