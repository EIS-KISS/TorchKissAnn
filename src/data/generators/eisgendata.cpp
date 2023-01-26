#include "eisgendata.h"

#include <cstddef>
#include <cstdint>
#include <vector>
#include <assert.h>
#include <math.h>
#include <cstdlib>
#include <eisgenerator/normalize.h>
#include <eisgenerator/basicmath.h>
#include <sstream>
#include <fstream>
#include <istream>

#include "../eistotorch.h"
#include "log.h"
#include "../print.h"
#include "../filterdata.h"
#include "tokenize.h"

#define INCBIN_PREFIX r
#include "incbin.h"

using namespace eis;

INCTXT(Circuits, "../circuits.txt");

static std::vector<std::string> readCircutsFromStream(std::istream& ss)
{
	std::vector<std::string> out;

	while(ss.good())
	{
		std::string line;
		std::getline(ss, line);
		if(!line.empty() && line[0] != '#')
			out.push_back(line);
	}

	return out;
}

EisGeneratorDataset::EisGeneratorDataset(int64_t outputSize, double noiseI):
omega(10, 1e6, outputSize/2, true), noise(noiseI)
{
}

EisGeneratorDataset::EisGeneratorDataset(std::istream& is, int64_t desiredSize, int64_t outputSize, double noise, bool inductivity):
EisGeneratorDataset(outputSize, noise)
{
	std::vector<std::string> circuits = readCircutsFromStream(is);
	if(circuits.empty())
		throw eis::file_error("stream dosent contain any circuits");
	addVectorOfModels(circuits, desiredSize, inductivity);
}

EisGeneratorDataset::EisGeneratorDataset(const std::filesystem::path& path, int64_t desiredSize, int64_t outputSize, double noise, bool inductivity):
EisGeneratorDataset(outputSize, noise)
{
	std::ifstream is(path, std::ios::in);
	if(!is.is_open())
		throw eis::file_error("can not open " + path.string());
	std::vector<std::string> circuits = readCircutsFromStream(is);
	if(circuits.empty())
		throw eis::file_error("file dosent contain any circuits: " + path.string());
	addVectorOfModels(circuits, desiredSize, inductivity);
}

EisGeneratorDataset::EisGeneratorDataset(const char* cStr, size_t cStrLen, int64_t desiredSize, int64_t outputSize, double noise, bool inductivity):
EisGeneratorDataset(outputSize, noise)
{
	std::stringstream ss(std::string(cStr, cStrLen));
	std::vector<std::string> circuits = readCircutsFromStream(ss);
	if(circuits.empty())
		throw std::runtime_error("array contains no circuits");
	addVectorOfModels(circuits, desiredSize, inductivity);
}

void EisGeneratorDataset::addVectorOfModels(const std::vector<std::string>& modelStrs, int64_t desiredSize, bool inductivity)
{
	int64_t sizePerModel = desiredSize/modelStrs.size();
	assert(sizePerModel > 3);
	for(const std::string& modelStr : modelStrs)
	{
		std::string workModelStr = stripWhitespace(modelStr);
		if(workModelStr.empty())
			continue;

		if(inductivity)
			workModelStr.insert(0, std::string("l{5e-7}-"));

		Log(Log::DEBUG)<<__func__<<" adding model "<<workModelStr;

		Model model(workModelStr);
		if(!model.isReady())
		{
			Log(Log::WARN)<<__func__<<" invalid model string "<<workModelStr;
			continue;
		}
		model.setParamSweepCountClosestTotal(sizePerModel);

		addModel(model);
	}

	if(desiredSize/2 > trueSize())
	{
		repetitionCount = desiredSize/trueSize();
	}
}

void EisGeneratorDataset::addModel(const eis::Model& model)
{
	models.push_back(model);
	modelSweepSizes.push_back(models.back().getRequiredStepsForSweeps());
	std::vector<std::string>::iterator iterator = std::find(modelStrings.begin(), modelStrings.end(), models.back().getModelStr());
	size_t index;
	if(iterator == modelStrings.end())
	{
		modelStrings.push_back(models.back().getModelStr());
		index = modelStrings.size()-1;
	}
	else
	{
		index = std::distance(modelStrings.begin(), iterator);
	}

	modelClasses.push_back(index);
}

std::pair<size_t, size_t> EisGeneratorDataset::getModelAndOffsetForIndex(size_t index) const
{
	size_t model = 0;
	for(; model < models.size(); ++model)
	{
		if(index < modelSweepSizes[model])
			break;
		else
			index -= modelSweepSizes[model];
	}

	return std::pair<size_t, size_t>(model, index);
}

torch::data::Example<torch::Tensor, torch::Tensor> EisGeneratorDataset::get(size_t index)
{
	index = index % trueSize();
	std::pair<size_t, size_t> modelAndOffset = getModelAndOffsetForIndex(index);
	std::vector<eis::DataPoint> data = models[modelAndOffset.first].executeSweep(omega, modelAndOffset.second);

	assert(data.size());
	filterData(data, omega.count*2);

	if(data.size() != omega.count)
	{
		if constexpr(PRINT)
			std::cout<<__func__<<' '<<index<<" rejected as uninteresting\n";
		return get(index+1);
	}

	if(noise > 0)
		eis::noise(data, noise, false);

	torch::Tensor input = eisToTorch(data);

	torch::TensorOptions options;
	options = options.layout(torch::kStrided);
	options = options.device(torch::kCPU);
	options = options.dtype(torch::kInt8);
	torch::Tensor output = torch::zeros({1}, options);
	output[0] = static_cast<float>(modelClasses[modelAndOffset.first]);

	return torch::data::Example<torch::Tensor, torch::Tensor>(input, output);
}

size_t EisGeneratorDataset::classesCount() const
{
	return modelStrings.size();
}

size_t EisGeneratorDataset::frequencies()
{
	return omega.count;
}

c10::optional<size_t> EisGeneratorDataset::size() const
{
	return trueSize()*repetitionCount;
}

size_t EisGeneratorDataset::trueSize() const
{
	size_t size = 0;
	for(size_t i = 0; i < models.size(); ++i)
		size += modelSweepSizes[i];
	return size;
}

torch::Tensor EisGeneratorDataset::classWeights()
{
	torch::TensorOptions options;
	options = options.dtype(torch::kFloat32);
	options = options.layout(torch::kStrided);
	options = options.device(torch::kCPU);
	torch::Tensor output = torch::zeros({static_cast<int64_t>(classesCount())}, options);
	int64_t count = trueSize();
	for(size_t i = 0; i < models.size(); ++i)
	{
		auto iterator = std::find(modelStrings.begin(), modelStrings.end(), models[i].getModelStr());
		assert(iterator != modelStrings.end());

		size_t classNum = std::distance(modelStrings.begin(), iterator);
		output[classNum] = output[classNum] + static_cast<int64_t>(modelSweepSizes[i]);
	}

	output = output/static_cast<double>(count);
	output = torch::min(output)/output;

	//output = torch::nn::functional::normalize(output, torch::nn::functional::NormalizeFuncOptions().p(2).dim(0));
	return output;
}

size_t EisGeneratorDataset::classForIndex(size_t index)
{
	std::pair<size_t, size_t> modelAndOffset = getModelAndOffsetForIndex(index);
	return modelClasses[modelAndOffset.first];
}

EisGeneratorDataset* EisGeneratorDataset::getTestDataset()
{
	return this;
}

EisGeneratorDataset* EisGeneratorDataset::newDefaultDataset(int64_t outputSize, int64_t desiredSize, double noise, bool inductivity)
{
	assert(outputSize > 2);
	assert(desiredSize > 1000);
	EisGeneratorDataset* dataset = new EisGeneratorDataset(rCircuitsData, rCircuitsSize, desiredSize, outputSize, noise, inductivity);

	Log(Log::DEBUG)<<__func__<<" with "<<dataset->size().value()<<" examples";
	return dataset;
}

std::string EisGeneratorDataset::modelStringForClass(size_t classNum)
{
	if(classNum >= modelStrings.size())
		return "invalid";
	else
		return modelStrings[classNum];
}
