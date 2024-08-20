#include "log.h"
#include "coincellhellloader.h"

#include <ATen/ops/maximum.h>
#include <cassert>
#include <cstdint>
#include <eisgenerator/eistype.h>
#include <eisgenerator/model.h>
#include <execution>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <vector>

#include "../eistotorch.h"
#include "randomgen.h"
#include "spectra.h"
#include "tensoroptions.h"
#include "loaderror.h"

CoinCellHellLoaderPostdict::CoinCellHellLoaderPostdict(const std::filesystem::path& pathI): CoinCellHellLoader(pathI)
{
}

std::vector<std::pair<std::string, int64_t>> CoinCellHellLoaderPostdict::extraInputs()
{
	return {{"ocv", 1}};
}

size_t CoinCellHellLoaderPostdict::outputSize() const
{
	return 5;
}


std::string CoinCellHellLoaderPostdict::modelStringForClass(size_t classNum)
{
	switch(classNum)
	{
		case 0:
			return "CellGroup";
		case 1:
			return "Temperature";
		case 2:
			return "ChargeCycles";
		case 3:
			return "ThermalCycles";
		case 4:
			return "SOC";
		default:
			return "CoincellHell";
	}
}

torch::data::Example<torch::Tensor, torch::Tensor> CoinCellHellLoaderPostdict::getImpl(size_t index)
{
	eis::EisSpectra spectra = loadSpectra(index);
	Metadata meta = loadMetadata(spectra);

	std::vector<eis::DataPoint> data = spectra.data;
	torch::Tensor input = eisToTorchExtra(data, {meta.ocv});

	torch::TensorOptions options;
	options = options.layout(torch::kStrided);
	options = options.device(torch::kCPU);
	options = options.dtype(torch::kFloat32);
	torch::Tensor output = torch::empty({static_cast<int64_t>(outputSize())}, options);

	output[0] = meta.cell_group;
	output[1] = meta.temparature;
	output[2] = meta.charge_cycles;
	output[3] = meta.thermal_cycles;
	output[4] = meta.soc > 0 ? meta.soc : meta.soc_estimate;

	return torch::data::Example<torch::Tensor, torch::Tensor>(input, output);
}

CoinCellHellLoaderPredict::CoinCellHellLoaderPredict(const std::filesystem::path& pathI): CoinCellHellLoader(pathI)
{
	std::unordered_map<int, std::vector<std::pair<eis::EisSpectra, Metadata>>> cellvecs;
	for(size_t i = 0; i < CoinCellHellLoader::size().value(); ++i)
	{
		eis::EisSpectra spectra = loadSpectra(i);
		Metadata meta = loadMetadata(spectra);
		cellvecs[meta.cellid].push_back({spectra, meta});
	}

	Log(Log::DEBUG)<<__func__<<" processing dataset with "<<CoinCellHellLoader::size().value()<<" examples and "<<cellvecs.size()<<" cells";

	for(std::pair<int, std::vector<std::pair<eis::EisSpectra, Metadata>>> cellvec : cellvecs)
	{
		std::sort(std::execution::par, cellvec.second.begin(), cellvec.second.end(), [](auto a, auto b){return a.second.step < b.second.step;});
		for(size_t i = 0; i < cellvec.second.size(); ++i)
		{
			if(cellvec.second[i].second.cap_estimate < 0)
				continue;
			int targetDeltaCy = rd::rand(100)+20;
			const std::pair<eis::EisSpectra, Metadata>* foundCounterpart = nullptr;
			for(size_t k = i+1; k < cellvec.second.size()-1; ++k)
			{
				if(cellvec.second[k].second.charge_cycles - cellvec.second[i].second.charge_cycles >= targetDeltaCy)
				{
					foundCounterpart = &(cellvec.second[k]);
					break;
				}
			}

			if(!foundCounterpart)
			{
				if(cellvec.second.back().second.charge_cycles - cellvec.second[i].second.charge_cycles >= 20)
					foundCounterpart = &cellvec.second.back();
			}

			if(foundCounterpart)
			{
				const Metadata &meta = foundCounterpart->second;
				const std::vector<eis::DataPoint>& data = foundCounterpart->first.data;
				torch::Tensor input = eisToTorchExtra(data, {meta.ocv, meta.temparature, (meta.cap_estimate - cellvec.second[i].second.cap_estimate),
					static_cast<float>(meta.charge_cycles -  cellvec.second[i].second.charge_cycles)});

				torch::Tensor output = torch::empty({static_cast<int64_t>(outputSize())}, tensorOptCpu<fvalue>(false));
				output[0] = (meta.cap_estimate - cellvec.second[i].second.cap_estimate);
				examples.push_back({input, output});
			}
		}
	}

	Log(Log::DEBUG)<<__func__<<" prepared "<<size().value()<<" examples";
}

std::vector<std::pair<std::string, int64_t>> CoinCellHellLoaderPredict::extraInputs()
{
	return {{"ocv", 1}, {"temperature", 1}, {"soc", 1}, {"delta_cycles", 1}};
}

size_t CoinCellHellLoaderPredict::outputSize() const
{
	return 1;
}

std::string CoinCellHellLoaderPredict::modelStringForClass(size_t classNum)
{
	switch(classNum)
	{
		case 0:
			return "DeltaCap";
		default:
			return "CoincellHell";
	}
}

c10::optional<size_t> CoinCellHellLoaderPredict::size() const
{
	return examples.size();
}

torch::data::Example<torch::Tensor, torch::Tensor> CoinCellHellLoaderPredict::getImpl(size_t index)
{
	return examples[index];
}
