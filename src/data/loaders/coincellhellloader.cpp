#include "log.h"
#include "coincellhellloader.h"

#include <ATen/ops/maximum.h>
#include <cassert>
#include <cstdint>
#include <eisgenerator/eistype.h>
#include <eisgenerator/model.h>
#include <sstream>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <vector>
#include <limits>

#include "../eistotorch.h"
#include "microtar.h"
#include "tensoroptions.h"
#include "loaderror.h"

CoinCellHellLoader::CoinCellHellLoader(const std::filesystem::path& pathI): path(pathI)
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
	if(files.size() > 0)
		getImpl(0);
}

CoinCellHellLoader::CoinCellHellLoader(const CoinCellHellLoader& in)
{
	operator=(in);
}

CoinCellHellLoader& CoinCellHellLoader::operator=(const CoinCellHellLoader& in)
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

CoinCellHellLoader::~CoinCellHellLoader()
{
	if(tar.stream)
		mtar_close(&tar);
}

std::string CoinCellHellLoader::dataLabel() const
{
	return "EIS";
}

eis::EisSpectra CoinCellHellLoader::loadSpectra(size_t index)
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

CoinCellHellLoader::Metadata CoinCellHellLoader::loadMetadata(const eis::EisSpectra& spectra)
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

torch::data::Example<torch::Tensor, torch::Tensor> CoinCellHellLoader::getImpl(size_t index)
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

size_t CoinCellHellLoader::outputSize() const
{
	return 5;
}

c10::optional<size_t> CoinCellHellLoader::size() const
{
	return files.size();
}

size_t CoinCellHellLoader::classForIndex(size_t index)
{
	(void)index;
	return 0;
}

std::string CoinCellHellLoader::modelStringForClass(size_t classNum)
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

bool CoinCellHellLoader::isMulticlass()
{
	return true;
}

c10::optional<torch::Tensor> CoinCellHellLoader::freqRange()
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


std::pair<torch::Tensor, torch::Tensor> CoinCellHellLoader::getTargetScalesAndBias()
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

std::vector<std::pair<std::string, int64_t>> CoinCellHellLoader::extraInputs()
{
	return {{"ocv", 1}};
}
