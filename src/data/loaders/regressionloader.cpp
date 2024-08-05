#include "regressionloader.h"

#include <cassert>
#include <eisgenerator/eistype.h>
#include <eisgenerator/model.h>
#include <sstream>

#include "microtar.h"
#include "../eistotorch.h"
#include "tensoroptions.h"
#include "tokenize.h"
#include "loaderror.h"

ParameterRegressionDatasetLoader::ParameterRegressionDatasetLoader(const std::filesystem::path& pathI, bool drtI): path(pathI), drt(drtI)
{
	parameterCount = 0;
	int ret = mtar_open(&tar, path.c_str(), "r");
	if(ret != 0)
	{
		Log(Log::WARN)<<path<<" is not a valid tar archive";
		return;
	}

	mtar_header_t header;
	while((mtar_read_header(&tar, &header)) != MTAR_ENULLRECORD)
	{
		if(header.type == MTAR_TREG)
		{
			std::filesystem::path path = header.name;
			files.push_back({path, header.size});
		}
		mtar_next(&tar);
	}
	if(files.size() > 0)
		getImpl(0);
}

ParameterRegressionDatasetLoader::ParameterRegressionDatasetLoader(const ParameterRegressionDatasetLoader& in)
{
	operator=(in);
}

ParameterRegressionDatasetLoader& ParameterRegressionDatasetLoader::operator=(const ParameterRegressionDatasetLoader& in)
{
	parameterCount = in.parameterCount;
	dataSizes = in.dataSizes;
	path = in.path;
	int ret = mtar_open(&tar, path.c_str(), "r");
	drt = in.drt;
	if(ret != 0)
	{
		Log(Log::ERROR)<<"Unable to reopen tar file at "<<path;
		assert(ret == 0);
	}
	files = in.files;
	std::vector<int64_t> dataSizes = {};

	return *this;
}

ParameterRegressionDatasetLoader::~ParameterRegressionDatasetLoader()
{
	mtar_close(&tar);
}

std::string ParameterRegressionDatasetLoader::dataLabel() const
{
	return "DRT";
}

eis::EisSpectra ParameterRegressionDatasetLoader::loadSpectra(size_t index)
{
	mtar_header_t header;
	int ret = mtar_find(&tar, files[index].first.c_str(), &header);
	if(ret != 0)
	{
		Log(Log::ERROR)<<"Unable to find "<<files[index].first<<" in tar archive";
		assert(ret == 0);
	}

	Log(Log::DEBUG)<<"Reading \""<<files[index].first.c_str()<<"\" of length "<<header.size<<'/'<<files[index].second<<" from tar";
	char* filebuffer = new char[header.size+1];
	filebuffer[header.size] = '\0';
	ret = mtar_read_data(&tar, filebuffer, header.size);
	if(ret != 0)
	{
		Log(Log::ERROR)<<"Unable to read "<<files[index].first<<" from tar archive";
		assert(ret == 0);
	}
	std::stringstream ss(filebuffer);

	eis::EisSpectra spectra = eis::EisSpectra::loadFromStream(ss);
	delete[] filebuffer;

	return spectra;
}

torch::data::Example<torch::Tensor, torch::Tensor> ParameterRegressionDatasetLoader::getImpl(size_t index)
{
	eis::EisSpectra spectra = loadSpectra(index);

	if(parameterCount == 0)
	{
		parameterCount = spectra.labels.size();
		if(parameterCount == 0)
			throw loader_error("Parameter count of file " + files[index].first.string() + " is 0");
	}
	else if(parameterCount != spectra.labels.size())
	{
		throw loader_error("Parameter count of all files must be the same at " + std::to_string(parameterCount)
			+ " however " + files[index].first.string() + " contains " + std::to_string(spectra.labels.size()));
	}

	torch::Tensor input = eisToTorch(spectra.data);

	if(drt)
		input = torch::split(input, input.numel()/2, 0)[0];

	torch::TensorOptions options;
	options = options.layout(torch::kStrided);
	options = options.device(torch::kCPU);
	options = options.dtype(torch::kFloat32);
	torch::Tensor output = torch::empty({static_cast<int64_t>(parameterCount)}, options);

	if(dataSizes.empty())
	{
		dataSizes = output.sizes().vec();
	}
	else
	{
		if(dataSizes.size() != output.sizes().size())
			throw loader_error("Dimmentions should be the same at " + std::to_string(dataSizes.size())
				+ " however " + files[index].first.string() + " contains "
				+ std::to_string(output.sizes().size()));

		for(size_t i = 0; i < dataSizes.size(); ++i)
		{
			if(dataSizes[i] != output.sizes()[i])
			{
				throw loader_error("Output dimention " + std::to_string(i)
				+ " is " + std::to_string(output.sizes()[i]) + " but "
				+ std::to_string(dataSizes[i]) + " is expected");
			}
		}
	}

	size_t i = 0;
	auto tensorAccessor = output.accessor<float, 1>();
	for(double label : spectra.labels)
	{
		tensorAccessor[i] = label;
		++i;
	}

	return torch::data::Example<torch::Tensor, torch::Tensor>(input, output);
}

size_t ParameterRegressionDatasetLoader::outputSize() const
{
	return parameterCount;
}

c10::optional<size_t> ParameterRegressionDatasetLoader::size() const
{
	return files.size();
}

size_t ParameterRegressionDatasetLoader::classForIndex(size_t index)
{
	(void)index;
	return 0;
}

std::string ParameterRegressionDatasetLoader::modelStringForClass(size_t classNum)
{
	mtar_header_t header;
	int ret = mtar_find(&tar, files[0].first.c_str(), &header);
	if(ret != 0)
		throw loader_error("Could not find file in tar file at " + files[0].first.string());

	char* filebuffer = new char[header.size+1];
	filebuffer[header.size] = '\0';
	ret = mtar_read_data(&tar, filebuffer, header.size);
	if(ret != 0)
		throw loader_error("Could to read tar header for file at " + files[0].first.string());
	std::stringstream ss(filebuffer);
	eis::EisSpectra spectra = eis::EisSpectra::loadFromStream(ss);

	delete[] filebuffer;

	std::string modelStr = tokenize(spectra.model, '+')[0];
	Log(Log::ERROR)<<modelStr;
	eis::Model model(modelStr);

	std::vector<eis::Componant*> componants = model.getFlatComponants();
	for(eis::Componant* componant : componants)
	{
		if(classNum >= componant->paramCount())
			classNum -= componant->paramCount();
		else
			return std::string(1, componant->getComponantChar()) + std::string("-") + std::to_string(classNum);
	}
	return model.getModelStr();
}

bool ParameterRegressionDatasetLoader::isMulticlass()
{
	return true;
}

c10::optional<torch::Tensor> ParameterRegressionDatasetLoader::freqRange()
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

std::pair<torch::Tensor, torch::Tensor> ParameterRegressionDatasetLoader::getTargetScalesAndBias()
{
	eis::Model model(modelStringForClass(outputSize()+1));
	std::vector<eis::Range> defaultRanges = model.getDefaultParameters();
	torch::Tensor scales = torch::empty({static_cast<int64_t>(defaultRanges.size())}, tensorOptCpu<fvalue>(false));
	torch::Tensor bias = torch::empty({static_cast<int64_t>(defaultRanges.size())}, tensorOptCpu<fvalue>(false));

	for(size_t i = 0; i < defaultRanges.size(); ++i) {
		scales[i] = defaultRanges[i].end-defaultRanges[i].start;
		bias[i] = defaultRanges[i].start;
	}
	return {scales, bias};
}
