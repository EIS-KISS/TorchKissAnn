#include "eisgendata.h"

#include <cstddef>
#include <cstdint>
#include <vector>
#include <assert.h>
#include <math.h>
#include <cstdlib>
#include <eisgenerator/normalize.h>
#include <eisgenerator/basicmath.h>

#include "../eistotorch.h"
#include "log.h"
#include "../print.h"
#include "../filterdata.h"

using namespace eis;

EisGeneratorDataset::EisGeneratorDataset(int64_t outputSize, double noiseI): omega(10, 1e6, outputSize/2, true), noise(noiseI)
{
}

void EisGeneratorDataset::addModel(const eis::Model& model, std::vector<eis::Range> modelRange)
{
	models.push_back(model);
	modelRanges.push_back(modelRange);
	modelSweepSizes.push_back(eis::Model::getRequiredStepsForSweeps(modelRange));

	Log(Log::DEBUG)<<"EisGeneratorDataset "<<__func__<<" for "<<model.getModelStr()<<" with "<<modelSweepSizes[models.size()-1]<<" sweeps";
	for(size_t i = 0; i < modelRange.size(); ++i)
	{
		modelRange[i].print(Log::ERROR);
	}
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

#if 0
	if(model == 5)
	{
		static int grand = 0;
		grand++;
		if(grand == 100 || grand == 1000)
		{
			Log(Log::ERROR)<<models[model].getModelStr()<<" i: "<<grand<<"\nomega,real,imag";
			printDataVect(data);
			if(grand == 1000)
				exit(1);
		}
	}
#endif


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

torch::Tensor EisGeneratorDataset::modelWeights()
{
	torch::TensorOptions options;
	options = options.dtype(torch::kFloat32);
	options = options.layout(torch::kStrided);
	options = options.device(torch::kCPU);
	torch::Tensor output = torch::empty({static_cast<int64_t>(modelCount())}, options);
	size_t count = size().value();
	for(size_t i = 0; i < modelCount(); ++i)
	{
		output[i] = count/static_cast<float>(Model::getRequiredStepsForSweeps(modelRanges[i]));
	}

	//output = torch::nn::functional::normalize(output, torch::nn::functional::NormalizeFuncOptions().p(2).dim(0));
	Log(Log::DEBUG)<<"Weights:";
	Log(Log::DEBUG)<<output;
	return output;
}

EisGeneratorDataset* EisGeneratorDataset::getTestDataset()
{
	return this;
	//EisGeneratorDataset* testDataset =
}

static std::vector<eis::Range> getRange(size_t index, size_t sweepDepth, bool inductivity)
{
	std::vector<eis::Range> modelRange;

	if(inductivity)
		modelRange.push_back(eis::Range(5e-7, 5e-7, 1, true, 'l'));

	switch(index)
	{
		case 0:
			modelRange.push_back(eis::Range(1e-7, 1e-5, sweepDepth, true, 'c'));
			modelRange.push_back(eis::Range(10, 100e3, sweepDepth, true, 'r'));
			modelRange.push_back(eis::Range(1e-7, 1e-5, sweepDepth, true, 'c'));
			modelRange.push_back(eis::Range(10, 100e3, sweepDepth, true, 'r'));
			modelRange.push_back(eis::Range(1e-7, 1e-5, sweepDepth, true, 'c'));
			modelRange.push_back(eis::Range(10, 100e3, sweepDepth, true, 'r'));
			break;
		case 1:
			modelRange.push_back(eis::Range(0.1, 10, sweepDepth*2, true, 'r'));
			modelRange.push_back(eis::Range(1e-6, 1e-4, sweepDepth*2, true, 'c'));
			modelRange.push_back(eis::Range(0.1, 10, sweepDepth*2, true, 'r'));
			break;
		case 2:
			modelRange.push_back(eis::Range(1e2, 1e4, sweepDepth, true, 'r'));
			modelRange.push_back(eis::Range(1e-9, 1e-4, sweepDepth, true, 'c'));
			modelRange.push_back(eis::Range(1e2, 1e4, sweepDepth, true, 'r'));
			modelRange.push_back(eis::Range(1e-9, 1e-4, sweepDepth, true, 'c'));
			modelRange.push_back(eis::Range(1e2, 1e4, sweepDepth, true, 'r'));
			modelRange.push_back(eis::Range(1e-9, 1e-4, sweepDepth, true, 'c'));
			break;
		case 3:
			modelRange.push_back(eis::Range(1e1, 1e3, sweepDepth, true, 'r'));
			modelRange.push_back(eis::Range(1e-5, 1e-3, sweepDepth, true, 'l'));
			modelRange.push_back(eis::Range(1e1, 1e3, sweepDepth, true, 'r'));
			modelRange.push_back(eis::Range(1e-9, 1e-5, sweepDepth, true, 'c'));
			modelRange.push_back(eis::Range(1e1, 1e3, sweepDepth, true, 'r'));
			break;
		case 4:
			modelRange.push_back(eis::Range(1e2, 1e4, sweepDepth, true, 'r'));
			modelRange.push_back(eis::Range(1e-7, 1e-5, sweepDepth, true, 'c'));
			modelRange.push_back(eis::Range(1e2, 1e4, sweepDepth, true, 'r'));
			modelRange.push_back(eis::Range(1e-10, 1e-7, sweepDepth, true, 'c'));
			modelRange.push_back(eis::Range(1e2, 1e4, sweepDepth, true, 'r'));
			break;
		case 5:
			modelRange.push_back(eis::Range(10, 1e4, sweepDepth, true, 'r'));
			modelRange.push_back(eis::Range(10, 1e4, sweepDepth, true, 'r'));
			modelRange.push_back(eis::Range(1e-12, 1e-5, sweepDepth, true, 'c'));
			modelRange.push_back(eis::Range(10, 1e4, sweepDepth, true, 'r'));
			modelRange.push_back(eis::Range(1e-12, 1e-5, sweepDepth, true, 'c'));
			modelRange.push_back(eis::Range(10, 1e4, sweepDepth, true, 'r'));
			modelRange.push_back(eis::Range(1e-12, 1e-5, sweepDepth, true, 'c'));
			break;
		case 6:
			modelRange.push_back(eis::Range(1e3, 1e4, sweepDepth*10, true, 't'));
			modelRange.push_back(eis::Range(1e-7, 1e-6, sweepDepth*10, true, 't'));
			modelRange.push_back(eis::Range(5, 5, 1, true, 't'));
		case 7:
			modelRange.push_back(eis::Range(10, 100, sweepDepth*100, true, 'w'));
			break;
	}
	return modelRange;
}

EisGeneratorDataset* EisGeneratorDataset::getNeisGeneratedDataset(int64_t outputSize, int sweepDepth, double noise, bool inductivity)
{
	EisGeneratorDataset* dataset = new EisGeneratorDataset(outputSize, noise);

	std::string modelStr;

	const std::string probeModelString("l{5e-7}-");

	modelStr.assign("(c{1e-6}-r{47})(c{1e-6}-r{1e3})(c{1e-6}-r{22e3})");
	if(inductivity)
		modelStr.insert(0, probeModelString);
	eis::Model bord1_1(modelStr);
	assert(bord1_1.checkParameterRanges(getRange(0, sweepDepth, inductivity)));
	dataset->addModel(bord1_1, getRange(0, sweepDepth, inductivity));

	modelStr.assign("r{0.5}-c{10e-6}r{0.5}");
	if(inductivity)
		modelStr.insert(0, probeModelString);
	eis::Model bord1_3(modelStr);
	assert(bord1_3.checkParameterRanges(getRange(1, sweepDepth, inductivity)));
	dataset->addModel(bord1_3, getRange(1, sweepDepth, inductivity));

	modelStr.assign("r{100}c{1e-6}-r{100}c{100e-9}-r{100}c{100e-12}");
	if(inductivity)
		modelStr.insert(0, probeModelString);
	eis::Model bord1_4(modelStr);
	assert(bord1_4.checkParameterRanges(getRange(2, sweepDepth, inductivity)));
	dataset->addModel(bord1_4, getRange(2, sweepDepth, inductivity));

	modelStr.assign("r{47}l{100e-6}-r{47}c{1e-6}-r{47}");
	if(inductivity)
		modelStr.insert(0, probeModelString);
	eis::Model bord1_5(modelStr);
	assert(bord1_5.checkParameterRanges(getRange(3, sweepDepth, inductivity)));
	dataset->addModel(bord1_5, getRange(3, sweepDepth, inductivity));

	modelStr.assign("r{1e3}c{1e-6}-r{1e3}c{1e-9}-r{47}");
	if(inductivity)
		modelStr.insert(0, probeModelString);
	eis::Model bord2(modelStr);
	assert(bord2.checkParameterRanges(getRange(4, sweepDepth, inductivity)));
	dataset->addModel(bord2, getRange(4, sweepDepth, inductivity));

//	modelStr.assign("r{100}-r{470}c{100e-12}-r{1e3}c{1e-9}-r{100}c{1e-6}");
//	if(inductivity)
//		modelStr.insert(0, probeModelString);
//	eis::Model bord4(modelStr);
//	assert(bord4.checkParameterRanges(getRange(5, sweepDepth, inductivity)));
//	dataset->addModel(bord4, getRange(5, sweepDepth, inductivity));

	modelStr.assign("t{1e3, 1e-7, 3}");
	if(inductivity)
		modelStr.insert(0, probeModelString);
	eis::Model bordTl(modelStr);
	assert(bordTl.checkParameterRanges(getRange(6, sweepDepth, inductivity)));
	dataset->addModel(bordTl, getRange(6, sweepDepth, inductivity));

	modelStr.assign("w{1000}");
	if(inductivity)
		modelStr.insert(0, probeModelString);
	eis::Model bordW(modelStr);
	assert(bordW.checkParameterRanges(getRange(7, sweepDepth, inductivity)));
	dataset->addModel(bordW, getRange(7, sweepDepth, inductivity));

	Log(Log::DEBUG)<<__func__<<" with "<<dataset->size().value()<<" examples";
	return dataset;
}

std::string EisGeneratorDataset::getModelStrFromClass(size_t classNum)
{
	if(classNum >= models.size())
		return "invalid";

	else return models[classNum].getModelStr();
}
