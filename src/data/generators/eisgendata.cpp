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

EisGeneratorDataset::EisGeneratorDataset(int64_t outputSize, double noiseI): omega(10, 1e6, outputSize/2, true), noise(noiseI)
{
}

void EisGeneratorDataset::addModel(const eis::Model& model)
{
	models.push_back(model);
	modelSweepSizes.push_back(models.back().getRequiredStepsForSweeps());
	std::pair<std::set<std::string>::iterator, bool> iteratorPair = modelStrings.insert(models.back().getModelStr());
	size_t index = std::distance(modelStrings.begin(), iteratorPair.first);
	modelClasses.push_back(index);
}

std::pair<eis::Model*, size_t> EisGeneratorDataset::getModelAndOffsetForIndex(size_t index)
{
	size_t model = 0;
	for(; model < models.size(); ++model)
	{
		if(index < modelSweepSizes[model])
		{
			break;
		}
		else
		{
			index -= modelSweepSizes[model];
		}
	}

	return std::pair<eis::Model*, size_t>(&models[model], index);
}

torch::data::Example<torch::Tensor, torch::Tensor> EisGeneratorDataset::get(size_t index)
{
	std::pair<eis::Model*, size_t> modelAndOffset = getModelAndOffsetForIndex(index);
	std::vector<eis::DataPoint> data = modelAndOffset.first->executeSweep(omega, modelAndOffset.second);

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
	output[0] = static_cast<float>(classForIndex(index));

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
	int64_t count = size().value();
	for(size_t i = 0; i < models.size(); ++i)
	{
		auto iterator = modelStrings.find(models[i].getModelStr());
		assert(iterator != modelStrings.end());

		size_t classNum = std::distance(modelStrings.begin(), iterator);
		output[classNum] = output[classNum] + static_cast<int64_t>(modelSweepSizes[i]);
	}

	output = count/output;

	output = torch::nn::functional::normalize(output, torch::nn::functional::NormalizeFuncOptions().p(2).dim(0));
	return output;
}

size_t EisGeneratorDataset::classForIndex(size_t index)
{
	std::pair<eis::Model*, size_t> modelAndOffset = getModelAndOffsetForIndex(index);
	auto iterator = modelStrings.find(modelAndOffset.first->getModelStr());
	assert(iterator != modelStrings.end());
	size_t classNum = std::distance(modelStrings.begin(), iterator);
	return classNum;
}

EisGeneratorDataset* EisGeneratorDataset::getTestDataset()
{
	return this;
}

EisGeneratorDataset* EisGeneratorDataset::getNeisGeneratedDataset(int64_t outputSize, double noise, bool inductivity, const std::filesystem::path* path)
{
	EisGeneratorDataset* dataset = new EisGeneratorDataset(outputSize, noise);

	std::vector<std::string> modelStrs;
	if(!path)
	{
		std::stringstream ss(std::string(rCircuitsData, rCircuitsSize));
		modelStrs = readCircutsFromStream(ss);
	}
	else
	{
		std::ifstream is(path->c_str(), std::ios::in | std::ios::binary | std::ios::ate);
		modelStrs = readCircutsFromStream(is);
	}

	for(const std::string& modelStr : modelStrs)
	{
		Log(Log::DEBUG)<<__func__<<"adding model "<<modelStr;
		Model model(modelStr);
		if(!model.isReady())
		{
			Log(Log::WARN)<<__func__<<" invalid model string "<<modelStr;
			continue;
		}
		dataset->addModel(model);
	}

	Log(Log::DEBUG)<<__func__<<" with "<<dataset->size().value()<<" examples";
	return dataset;
}

std::string EisGeneratorDataset::modelStringForClass(size_t classNum)
{
	if(classNum >= modelStrings.size())
		return "invalid";
	else
		return *std::next(modelStrings.begin(), classNum);
}
