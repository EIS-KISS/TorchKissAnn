#include "eisgeneratorloader.h"

#include <cstddef>
#include <vector>
#include <assert.h>
#include <math.h>
#include <eisgenerator/normalize.h>

using namespace eis;

void printDataVect(const std::vector<eis::DataPoint> in)
{
	for(const eis::DataPoint& res : in)
			std::cout<<res.omega<<','<<res.im.real()<<','<<res.im.imag()<<'\n';
}

EisGeneratorDataset::EisGeneratorDataset(): omega(10, 1e6, OUTPUT_SIZE/2, true)
{
	models.push_back(Model("w{20e3}p{1e-7, 0.9}r{1e2}"));
	std::vector<eis::Range> modelRange;
	modelRange.push_back(eis::Range(1e3, 50e3, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e-7, 20e-7, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(0.7, 1.2, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e1, 1e3, 1));
	modelRanges.push_back(modelRange);
	modelSweepSizes.push_back(
		Model::getRequiredStepsForSweeps(modelRange));

	models.push_back(Model("c{20e-9}r{1e6}"));
	modelRange.clear();
	modelRange.push_back(eis::Range(1e-6, 1e-5, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e3, 20e3, SWEEP_DEPTH, true));
	modelRanges.push_back(modelRange);
	modelSweepSizes.push_back(
		Model::getRequiredStepsForSweeps(modelRange));

	models.push_back(Model("r{1e6}-c{20e-9}r{1e6}"));
	modelRange.clear();
	modelRange.push_back(eis::Range(1e3, 20e3, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e-6, 1e-5, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e3, 20e3, SWEEP_DEPTH, true));
	modelRanges.push_back(modelRange);
	modelSweepSizes.push_back(
	Model::getRequiredStepsForSweeps(modelRange));
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

	data = eis::reduceRegion(data);

	if constexpr(PRINT)
	{
		std::cout<<"Data reduced size "<<data.size()<<" data:\n";
		printDataVect(data);
	}

	if(data.size() < OUTPUT_SIZE/8)
	{
		if constexpr(PRINT)
			std::cout<<__func__<<' '<<index<<" rejected as uninteresting\n";
		return get(index+1);
	}
	data = eis::rescale(data, OUTPUT_SIZE/2);

	if constexpr(PRINT)
	{
		std::cout<<"Data rescaled size "<<data.size()<<" data:\n";
		printDataVect(data);
		assert(data.size()*2 == OUTPUT_SIZE);
	}

	torch::TensorOptions options;
	options = options.dtype(torch::kFloat32);
	options = options.layout(torch::kStrided);
	options = options.device(torch::kCPU);
	torch::Tensor input = torch::empty({static_cast<long int>(OUTPUT_SIZE)}, options);

	for(size_t i = 0; i < OUTPUT_SIZE; ++i)
	{
		float datapoint = i < data.size() ? data[i].im.real() : data[i - data.size()].im.imag();
		if(isnan(datapoint) || isinf(datapoint))
			datapoint = 0;
		input[i] = datapoint;
	}

	//torch::Tensor output = torch::zeros({static_cast<long int>(models.size())}, options);
	//output[model] = 1;

	options = options.dtype(torch::kInt64);
	torch::Tensor output = torch::zeros({1}, options);;
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
