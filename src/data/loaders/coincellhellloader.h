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

#include <eisgenerator/eistype.h>
#include <eisgenerator/model.h>
#include <string>
#include <filesystem>

#include "data/regressiondataset.h"
#include "microtar.h"
#include "tensoroptions.h"

template <typename DataSelf>
class CoinCellHellLoader :
public RegressionDataset<DataSelf>
{
protected:
	struct Metadata
	{
		long step;
		int substep;
		int cellid;
		int cell_group;
		float temparature;
		float ocv;
		int charge_cycles;
		int thermal_cycles;
		float last_avg_cap;
		long last_avg_step;
		float last_cap;
		long last_cap_step;
		float soc;
		float soc_estimate;
		float cap_estimate;
	};

private:

	std::vector<mtar_header_t> files;
	std::filesystem::path path;
	mtar_t tar = {};
	std::vector<int64_t> dataSizes = {};

protected:
	eis::Spectra loadSpectra(size_t index);
	Metadata loadMetadata(const eis::EisSpectra& spectra);
	virtual torch::data::Example<torch::Tensor, torch::Tensor> getImpl(size_t index) override = 0;

public:
	explicit CoinCellHellLoader(const std::filesystem::path& pathI);
	CoinCellHellLoader(const CoinCellHellLoader& in);
	CoinCellHellLoader& operator=(const CoinCellHellLoader& in);
	~CoinCellHellLoader();

	virtual c10::optional<size_t> size() const override;
	virtual bool isMulticlass() override;
	virtual std::string dataLabel() const override;
	virtual c10::optional<torch::Tensor> freqRange() override;
	virtual std::pair<torch::Tensor, torch::Tensor> getTargetScalesAndBias() override;
	virtual size_t outputSize() const override = 0;
	virtual size_t classForIndex(size_t index) override;
};

class CoinCellHellLoaderPostdict :
public CoinCellHellLoader<CoinCellHellLoaderPostdict>
{
protected:
	virtual torch::data::Example<torch::Tensor, torch::Tensor> getImpl(size_t index) override;

public:
	explicit CoinCellHellLoaderPostdict(const std::filesystem::path& pathI);
	virtual size_t outputSize() const override;
	virtual std::string modelStringForClass(size_t classNum) override;
	virtual std::vector<std::pair<std::string, int64_t>> extraInputs() override;
};

class CoinCellHellLoaderPredict :
public CoinCellHellLoader<CoinCellHellLoaderPostdict>
{
	std::vector<torch::data::Example<torch::Tensor, torch::Tensor>> examples;

protected:
	virtual torch::data::Example<torch::Tensor, torch::Tensor> getImpl(size_t index) override;

public:
	explicit CoinCellHellLoaderPredict(const std::filesystem::path& pathI);
	virtual size_t outputSize() const override;
	virtual c10::optional<size_t> size() const override;
	virtual std::string modelStringForClass(size_t classNum) override;
	virtual std::vector<std::pair<std::string, int64_t>> extraInputs() override;
};

template <typename DataSelf>
CoinCellHellLoader<DataSelf>::CoinCellHellLoader(const std::filesystem::path& pathI): path(pathI)
{
	int ret = mtar_open(&tar, path.c_str(), "r");
	if(ret != 0)
	{
		Log(Log::WARN)<<path<<" is not a valid tar archive";
		return;
	}

	mtar_header_t header;
	std::filesystem::path prevPath;
	while(mtar_read_header(&tar, &header) != MTAR_ENULLRECORD)
	{
		if(header.type == MTAR_TREG)
		{
			std::filesystem::path path = header.name;
			files.push_back(header);
			prevPath = path;
		}
		ret = mtar_next(&tar);
		if(ret != MTAR_ESUCCESS)
		{
			Log(Log::WARN)<<__func__<<": Unexpected end of tar file\n";
			break;
		}
	}
}

template <typename DataSelf>
CoinCellHellLoader<DataSelf>::CoinCellHellLoader(const CoinCellHellLoader& in)
{
	operator=(in);
}

template <typename DataSelf>
CoinCellHellLoader<DataSelf>& CoinCellHellLoader<DataSelf>::operator=(const CoinCellHellLoader& in)
{
	dataSizes = in.dataSizes;
	path = in.path;
	int ret = mtar_open(&tar, path.c_str(), "r");
	if(ret != 0)
	{
		Log(Log::ERROR)<<"Unable to reopen tar file at "<<path;
		assert(ret == 0);
	}
	files = in.files;
	std::vector<int64_t> dataSizes = {};

	return *this;
}

template <typename DataSelf>
CoinCellHellLoader<DataSelf>::~CoinCellHellLoader()
{
	if(tar.stream)
		mtar_close(&tar);
}

template <typename DataSelf>
std::string CoinCellHellLoader<DataSelf>::dataLabel() const
{
	return "EIS";
}

template <typename DataSelf>
eis::EisSpectra CoinCellHellLoader<DataSelf>::loadSpectra(size_t index)
{
	Log(Log::DEBUG)<<"Reading \""<<files[index].name<<"\" of length "<<files[index].size<<" from tar";
	char* filebuffer = new char[files[index].size+1];
	filebuffer[files[index].size] = '\0';
	int ret = mtar_seek(&tar, files[index].pos);
	ret = mtar_read_data(&tar, filebuffer, files[index].size);
	if(ret != 0)
	{
		Log(Log::ERROR)<<"Unable to read "<<files[index].name<<" from tar archive";
		assert(ret == 0);
	}
	std::stringstream ss(filebuffer);

	eis::EisSpectra spectra = eis::EisSpectra::loadFromStream(ss);
	delete[] filebuffer;

	return spectra;
}

template <typename DataSelf>
typename CoinCellHellLoader<DataSelf>::Metadata CoinCellHellLoader<DataSelf>::loadMetadata(const eis::EisSpectra& spectra)
{
	Metadata metadata = {};
	for(size_t i = 0; i < spectra.labelNames.size(); ++i)
	{
		if(spectra.labelNames[i].find("substep") != std::string::npos)
			metadata.substep = spectra.labels[i];
		else if(spectra.labelNames[i].find("last_avg_step") != std::string::npos)
			metadata.last_avg_step = spectra.labels[i];
		else if(spectra.labelNames[i].find("last_cap_step") != std::string::npos)
			metadata.last_cap_step = spectra.labels[i];
		else if(spectra.labelNames[i].find("step") != std::string::npos)
			metadata.step = spectra.labels[i];
		else if(spectra.labelNames[i].find("cell_group") != std::string::npos)
			metadata.cell_group = spectra.labels[i];
		else if(spectra.labelNames[i].find("cellid") != std::string::npos)
			metadata.cellid = spectra.labels[i];
		else if(spectra.labelNames[i].find("temparature") != std::string::npos)
			metadata.temparature = spectra.labels[i];
		else if(spectra.labelNames[i].find("ocv") != std::string::npos)
			metadata.ocv = spectra.labels[i];
		else if(spectra.labelNames[i].find("charge_cycles") != std::string::npos)
			metadata.charge_cycles = spectra.labels[i];
		else if(spectra.labelNames[i].find("thermal_cycles") != std::string::npos)
			metadata.thermal_cycles = spectra.labels[i];
		else if(spectra.labelNames[i].find("last_avg_cap") != std::string::npos)
			metadata.last_avg_cap = spectra.labels[i];
		else if(spectra.labelNames[i].find("last_cap") != std::string::npos)
			metadata.last_cap = spectra.labels[i];
		else if(spectra.labelNames[i].find("soc_estimate") != std::string::npos)
			metadata.soc_estimate = spectra.labels[i];
		else if(spectra.labelNames[i].find("soc") != std::string::npos)
			metadata.soc = spectra.labels[i];
		else if(spectra.labelNames[i].find("cap_estimate") != std::string::npos)
			metadata.cap_estimate = spectra.labels[i];
		else
			Log(Log::WARN)<<__func__<<": unkown metadata label name '"<<spectra.labelNames[i]<<'\'';
	}
	return metadata;
}

template <typename DataSelf>
c10::optional<size_t> CoinCellHellLoader<DataSelf>::size() const
{
	return files.size();
}

template <typename DataSelf>
size_t CoinCellHellLoader<DataSelf>::classForIndex(size_t index)
{
	(void)index;
	return 0;
}

template <typename DataSelf>
bool CoinCellHellLoader<DataSelf>::isMulticlass()
{
	return true;
}

template <typename DataSelf>
c10::optional<torch::Tensor> CoinCellHellLoader<DataSelf>::freqRange()
{
	eis::EisSpectra spectra = loadSpectra(0);
	fvalue front = spectra.data.front().omega;
	fvalue back = spectra.data.back().omega;

	torch::Tensor out = torch::zeros({3}, tensorOptCpu<fvalue>(false));

	out[0] = std::min(front, back);
	out[1] = std::max(front, back);
	out[2] = static_cast<fvalue>(spectra.data.size());
	return out;
}

template <typename DataSelf>
std::pair<torch::Tensor, torch::Tensor> CoinCellHellLoader<DataSelf>::getTargetScalesAndBias()
{
	torch::Tensor max = torch::zeros({static_cast<int>(outputSize())}, tensorOptCpu<fvalue>(false));
	torch::Tensor min = torch::full({static_cast<int>(outputSize())}, std::numeric_limits<fvalue>::max(),  tensorOptCpu<fvalue>(false));

	for(size_t i = 0; i < size().value(); ++i)
	{
		torch::Tensor targets = getImpl(i).target;
		max = torch::maximum(targets, max);
		min = torch::minimum(targets, min);
	}
	return {(max-min), min};
}
