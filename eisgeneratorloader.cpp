#include "eisgeneratorloader.h"

#include <cstddef>
#include <vector>
#include <assert.h>
#include <math.h>
#include <eisgenerator/normalize.h>
#include <eisgenerator/basicmath.h>

#include "eistotorch.h"

using namespace eis;

void printDataVect(const std::vector<eis::DataPoint> in)
{
	for(const eis::DataPoint& res : in)
			std::cout<<res.omega<<','<<res.im.real()<<','<<res.im.imag()<<'\n';
}

EisGeneratorDataset::EisGeneratorDataset(int64_t outputSize): omega(10, 1e6, outputSize/2, true)
{
}

void EisGeneratorDataset::addModel(const eis::Model& model, std::vector<eis::Range> modelRange)
{
	models.push_back(model);
	modelRanges.push_back(modelRange);
	modelSweepSizes.push_back(eis::Model::getRequiredStepsForSweeps(modelRange));
}

torch::data::Example<torch::Tensor, torch::Tensor> EisGeneratorDataset::get(size_t index)
{
	std::vector<eis::DataPoint> data;
	size_t model = 0;
	for(; model < models.size(); ++model)
	{
		if(index < modelSweepSizes[model])
		{
			data = models[model].executeParamByIndex(modelRanges[model], omega, index);
			break;
		}
		else
		{
			index -= modelSweepSizes[model];
		}
	}

	assert(data.size());

	if constexpr(PRINT)
	{
		std::cout<<"Data as reported by lib size: "<<data.size()<<" data:\n";
		printDataVect(data);
	}

	filterData(data, omega.count*2);

	if(data.size() != omega.count)
	{
		if constexpr(PRINT)
			std::cout<<__func__<<' '<<index<<" rejected as uninteresting\n";
		return get(index+1);
	}

	torch::Tensor input = eisToTorch(data);

	torch::TensorOptions options;
	options = options.dtype(torch::kFloat32);
	options = options.layout(torch::kStrided);
	options = options.device(torch::kCPU);
	options = options.dtype(torch::kInt64);
	torch::Tensor output = torch::zeros({1}, options);
	output[0] = static_cast<float>(model);

	return torch::data::Example<torch::Tensor, torch::Tensor>(input, output);
}

size_t EisGeneratorDataset::modelCount()
{
	return models.size();
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

void EisGeneratorDataset::filterData(std::vector<eis::DataPoint>& data, size_t outputSize)
{
	data = eis::reduceRegion(data);

	if constexpr(PRINT)
	{
		std::cout<<"Data reduced size "<<data.size()<<" data:\n";
		printDataVect(data);
	}

	if(data.size() < outputSize/8)
	{
		data = std::vector<eis::DataPoint>();
		return;
	}
	data = eis::rescale(data, outputSize/2);

	if constexpr(PRINT)
	{
		std::cout<<"Data rescaled size "<<data.size()<<" data:\n";
		printDataVect(data);
		assert(data.size() == outputSize/2);
	}
}
