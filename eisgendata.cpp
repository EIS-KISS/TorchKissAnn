#include "eisgendata.h"

#include <cstddef>
#include <vector>
#include <assert.h>
#include <math.h>
#include <eisgenerator/normalize.h>
#include <eisgenerator/basicmath.h>

#include "eistotorch.h"
#include "log.h"

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

EisGeneratorDataset* EisGeneratorDataset::getNeisGeneratedDataset()
{
	EisGeneratorDataset* dataset = new EisGeneratorDataset();

	std::vector<eis::Range> modelRange;

	eis::Model bord1_1("(c{1e-6}-r{47})(c{1e-6}-r{1e3})(c{1e-6}-r{22e3})");

	modelRange.push_back(eis::Range(1e-7, 1e-5, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(10, 100e3, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e-7, 1e-5, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(10, 100e3, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e-7, 1e-5, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(10, 100e3, SWEEP_DEPTH, true));
	dataset->addModel(bord1_1, modelRange);

	eis::Model bord1_3("r{1e9}-c{1e-6}r{1e9}");
	modelRange.clear();
	modelRange.push_back(eis::Range(1e3, 1e10, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e-7, 1e-5, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e3, 1e10, SWEEP_DEPTH, true));
	dataset->addModel(bord1_3, modelRange);

	eis::Model bord1_4("r{100}c{1e-6}-r{100}c{100e-9}-r{100}c{100e-12}");
	modelRange.clear();
	modelRange.push_back(eis::Range(1e1, 1e4, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e-12, 1e-6, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e1, 1e4, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e-12, 1e-6, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e1, 1e4, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e-12, 1e-6, SWEEP_DEPTH, true));
	dataset->addModel(bord1_4, modelRange);

	eis::Model bord1_5("r{47}l{100e-6}-r{47}c{1e-6}-r{47}");
	modelRange.clear();
	modelRange.push_back(eis::Range(1e1, 1e3, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e-6, 1e-3, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e1, 1e3, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e-9, 1e-5, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e1, 1e3, SWEEP_DEPTH, true));
	dataset->addModel(bord1_5, modelRange);

	eis::Model bord2("r{1e3}c{1e-6}-r{1e3}c{1e-9}-r{47}");
	modelRange.clear();
	modelRange.push_back(eis::Range(1e1, 1e4, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e-12, 1e-5, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e1, 1e4, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e-12, 1e-5, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e1, 1e4, SWEEP_DEPTH, true));
	dataset->addModel(bord2, modelRange);

	eis::Model bord4("r{100}-r{470}c{100e-12}-r{1e3}c{1e-9}-r{100}c{1e-6}");
	modelRange.clear();
	modelRange.push_back(eis::Range(10, 1e4, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(10, 1e4, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e-12, 1e-5, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(10, 1e4, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e-12, 1e-5, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(10, 1e4, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e-12, 1e-5, SWEEP_DEPTH, true));
	dataset->addModel(bord4, modelRange);

	std::cout<<__func__<<" with "<<dataset->size().value()<<" examples\n";
	return dataset;
}


EisGeneratorDataset* EisGeneratorDataset::getNeisGeneratedDatasetAlt()
{
	EisGeneratorDataset* dataset = new EisGeneratorDataset();

	std::vector<eis::Range> modelRange;

	eis::Model bord1_1("(c{1e-6}-r{47})(c{1e-6}-r{1e3})(c{1e-6}-r{22e3})");
	modelRange.push_back(eis::Range(1e-7, 1e-5, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(10, 100e3, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(1e-7, 1e-5, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(10, 100e3, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(1e-7, 1e-5, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(10, 100e3, SWEEP_DEPTH/2, true));
	dataset->addModel(bord1_1, modelRange);

	eis::Model bord1_2("r{1e9}-c{1e-6}r{1e9}");
	modelRange.clear();
	modelRange.push_back(eis::Range(1e3, 1e10, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(1e-7, 1e-5, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(1e3, 1e10, SWEEP_DEPTH/2, true));
	dataset->addModel(bord1_2, modelRange);

	eis::Model bord1_4("r{100}c{1e-6}-r{100}c{100e-9}-r{100}c{100e-12}");
	modelRange.clear();
	modelRange.push_back(eis::Range(1e1, 1e4, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(1e-12, 1e-6, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(1e1, 1e4, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(1e-12, 1e-6, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(1e1, 1e4, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(1e-12, 1e-6, SWEEP_DEPTH/2, true));
	dataset->addModel(bord1_4, modelRange);

	eis::Model bord1_5("r{47}l{100e-6}-r{47}c{1e-6}-r{47}");
	modelRange.clear();
	modelRange.push_back(eis::Range(1e1, 1e3, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(1e-6, 1e-3, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(1e1, 1e3, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(1e-9, 1e-5, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(1e1, 1e3, SWEEP_DEPTH/2, true));
	dataset->addModel(bord1_5, modelRange);

	eis::Model bord2("r{1e3}c{1e-6}-r{1e3}c{1e-9}-r{47}");
	modelRange.clear();
	modelRange.push_back(eis::Range(1e1, 1e4, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(1e-12, 1e-5, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(1e1, 1e4, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(1e-12, 1e-5, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(1e1, 1e4, SWEEP_DEPTH/2, true));
	dataset->addModel(bord2, modelRange);

	eis::Model bord4("r{100}-r{470}c{100e-12}-r{1e3}c{1e-9}-r{100}c{1e-6}");
	modelRange.clear();
	modelRange.push_back(eis::Range(10, 1e4, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(10, 1e4, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(1e-12, 1e-5, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(10, 1e4, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(1e-12, 1e-5, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(10, 1e4, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(1e-12, 1e-5, SWEEP_DEPTH/2, true));
	dataset->addModel(bord4, modelRange);

	std::cout<<__func__<<" with "<<dataset->size().value()<<" examples\n";
	return dataset;
}

std::string EisGeneratorDataset::getModelStrFromClass(size_t classNum)
{
	if(classNum >= models.size())
		return "invalid";

	else return models[classNum].getModelStr();
}
