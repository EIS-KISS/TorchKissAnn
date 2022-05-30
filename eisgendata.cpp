#include "eisgendata.h"

#include <cstddef>
#include <cstdint>
#include <vector>
#include <assert.h>
#include <math.h>
#include <cstdlib>
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
/*
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
*/
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

EisGeneratorDataset* EisGeneratorDataset::getNeisGeneratedDataset(int64_t outputSize)
{
	EisGeneratorDataset* dataset = new EisGeneratorDataset(outputSize);

	std::vector<eis::Range> modelRange;

	eis::Model bord1_1("(c{1e-6}-r{47})(c{1e-6}-r{1e3})(c{1e-6}-r{22e3})");
	modelRange.push_back(eis::Range(1e-7, 1e-5, SWEEP_DEPTH, true, 'c'));
	modelRange.push_back(eis::Range(10, 100e3, SWEEP_DEPTH, true, 'r'));
	modelRange.push_back(eis::Range(1e-7, 1e-5, SWEEP_DEPTH, true, 'c'));
	modelRange.push_back(eis::Range(10, 100e3, SWEEP_DEPTH, true, 'r'));
	modelRange.push_back(eis::Range(1e-7, 1e-5, SWEEP_DEPTH, true, 'c'));
	modelRange.push_back(eis::Range(10, 100e3, SWEEP_DEPTH, true, 'r'));
	assert(bord1_1.checkParameterRanges(modelRange));
	dataset->addModel(bord1_1, modelRange);

	eis::Model bord1_3("r{0.5}-c{10e-6}r{0.5}");
	modelRange.clear();
	modelRange.push_back(eis::Range(0.1, 10, SWEEP_DEPTH, true, 'r'));
	modelRange.push_back(eis::Range(1e-6, 1e-4, SWEEP_DEPTH, true, 'c'));
	modelRange.push_back(eis::Range(0.1, 10, SWEEP_DEPTH, true, 'r'));
	assert(bord1_3.checkParameterRanges(modelRange));
	dataset->addModel(bord1_3, modelRange);

	eis::Model bord1_4("r{100}c{1e-6}-r{100}c{100e-9}-r{100}c{100e-12}");
	modelRange.clear();
	modelRange.push_back(eis::Range(1e1, 1e4, SWEEP_DEPTH, true, 'r'));
	modelRange.push_back(eis::Range(1e-12, 1e-6, SWEEP_DEPTH, true, 'c'));
	modelRange.push_back(eis::Range(1e1, 1e4, SWEEP_DEPTH, true, 'r'));
	modelRange.push_back(eis::Range(1e-12, 1e-6, SWEEP_DEPTH, true, 'c'));
	modelRange.push_back(eis::Range(1e1, 1e4, SWEEP_DEPTH, true, 'r'));
	modelRange.push_back(eis::Range(1e-12, 1e-6, SWEEP_DEPTH, true, 'c'));
	assert(bord1_4.checkParameterRanges(modelRange));
	dataset->addModel(bord1_4, modelRange);

	eis::Model bord1_5("r{47}l{100e-6}-r{47}c{1e-6}-r{47}");
	modelRange.clear();
	modelRange.push_back(eis::Range(1e1, 1e3, SWEEP_DEPTH, true, 'r'));
	modelRange.push_back(eis::Range(1e-6, 1e-3, SWEEP_DEPTH, true, 'l'));
	modelRange.push_back(eis::Range(1e1, 1e3, SWEEP_DEPTH, true, 'r'));
	modelRange.push_back(eis::Range(1e-9, 1e-5, SWEEP_DEPTH, true, 'c'));
	modelRange.push_back(eis::Range(1e1, 1e3, SWEEP_DEPTH, true, 'r'));
	assert(bord1_5.checkParameterRanges(modelRange));
	dataset->addModel(bord1_5, modelRange);

	eis::Model bord2("r{1e3}c{1e-6}-r{1e3}c{1e-9}-r{47}");
	modelRange.clear();
	modelRange.push_back(eis::Range(1e1, 1e4, SWEEP_DEPTH, true, 'r'));
	modelRange.push_back(eis::Range(1e-12, 1e-5, SWEEP_DEPTH, true, 'c'));
	modelRange.push_back(eis::Range(1e1, 1e4, SWEEP_DEPTH, true, 'r'));
	modelRange.push_back(eis::Range(1e-12, 1e-5, SWEEP_DEPTH, true, 'c'));
	modelRange.push_back(eis::Range(1e1, 1e4, SWEEP_DEPTH, true, 'r'));
	assert(bord2.checkParameterRanges(modelRange));
	dataset->addModel(bord2, modelRange);

//	eis::Model bord4("r{100}-r{470}c{100e-12}-r{1e3}c{1e-9}-r{100}c{1e-6}");
//	modelRange.clear();
//	modelRange.push_back(eis::Range(10, 1e4, SWEEP_DEPTH, true, 'r'));
//	modelRange.push_back(eis::Range(10, 1e4, SWEEP_DEPTH, true, 'r'));
//	modelRange.push_back(eis::Range(1e-12, 1e-5, SWEEP_DEPTH, true, 'c'));
//	modelRange.push_back(eis::Range(10, 1e4, SWEEP_DEPTH, true, 'r'));
//	modelRange.push_back(eis::Range(1e-12, 1e-5, SWEEP_DEPTH, true, 'c'));
//	modelRange.push_back(eis::Range(10, 1e4, SWEEP_DEPTH, true, 'r'));
//	modelRange.push_back(eis::Range(1e-12, 1e-5, SWEEP_DEPTH, true, 'c'));
//	assert(bord4.checkParameterRanges(modelRange));
//	dataset->addModel(bord4, modelRange);

	eis::Model bordTl("t{1e3, 1e-7, 3}");
	modelRange.clear();
	modelRange.push_back(eis::Range(1e2, 1e4, SWEEP_DEPTH*10, true, 't'));
	modelRange.push_back(eis::Range(1e-9, 1e-5, SWEEP_DEPTH*10, true, 't'));
	modelRange.push_back(eis::Range(3, 3, 1, true, 't'));
	assert(bordTl.checkParameterRanges(modelRange));
	dataset->addModel(bordTl, modelRange);

	Log(Log::DEBUG)<<__func__<<" with "<<dataset->size().value()<<" examples";
	return dataset;
}

EisGeneratorDataset* EisGeneratorDataset::getNeisGeneratedDatasetWithBLl(int64_t outputSize)
{
	EisGeneratorDataset* dataset = new EisGeneratorDataset(outputSize);

	std::vector<eis::Range> modelRange;

	eis::Model bord1_1("l{1e-7}-(c{1e-6}-r{47})(c{1e-6}-r{1e3})(c{1e-6}-r{22e3})");
	modelRange.push_back(eis::Range(5e-7, 5e-7, 1, true, 'l'));
	modelRange.push_back(eis::Range(1e-7, 1e-5, SWEEP_DEPTH, true, 'c'));
	modelRange.push_back(eis::Range(10, 100e3, SWEEP_DEPTH, true, 'r'));
	modelRange.push_back(eis::Range(1e-7, 1e-5, SWEEP_DEPTH, true, 'c'));
	modelRange.push_back(eis::Range(10, 100e3, SWEEP_DEPTH, true, 'r'));
	modelRange.push_back(eis::Range(1e-7, 1e-5, SWEEP_DEPTH, true, 'c'));
	modelRange.push_back(eis::Range(10, 100e3, SWEEP_DEPTH, true, 'r'));
	assert(bord1_1.checkParameterRanges(modelRange));
	dataset->addModel(bord1_1, modelRange);

	eis::Model bord1_3("l{5e-7}-r{0.5}-c{10e-6}r{0.5}");
	modelRange.clear();
	modelRange.push_back(eis::Range(5e-7, 5e-7, 1, true, 'l'));
	modelRange.push_back(eis::Range(0.1, 10, SWEEP_DEPTH*2, true, 'r'));
	modelRange.push_back(eis::Range(1e-6, 1e-4, SWEEP_DEPTH*2, true, 'c'));
	modelRange.push_back(eis::Range(0.1, 10, SWEEP_DEPTH*2, true, 'r'));
	assert(bord1_3.checkParameterRanges(modelRange));
	dataset->addModel(bord1_3, modelRange);

	eis::Model bord1_4("l{1e-7}-r{100}c{1e-6}-r{100}c{100e-9}-r{100}c{100e-12}");
	modelRange.clear();
	modelRange.push_back(eis::Range(5e-7, 5e-7, 1, true, 'l'));
	modelRange.push_back(eis::Range(1e1, 1e4, SWEEP_DEPTH, true, 'r'));
	modelRange.push_back(eis::Range(1e-12, 1e-6, SWEEP_DEPTH, true, 'c'));
	modelRange.push_back(eis::Range(1e1, 1e4, SWEEP_DEPTH, true, 'r'));
	modelRange.push_back(eis::Range(1e-12, 1e-6, SWEEP_DEPTH, true, 'c'));
	modelRange.push_back(eis::Range(1e1, 1e4, SWEEP_DEPTH, true, 'r'));
	modelRange.push_back(eis::Range(1e-12, 1e-6, SWEEP_DEPTH, true, 'c'));
	assert(bord1_4.checkParameterRanges(modelRange));
	dataset->addModel(bord1_4, modelRange);

	eis::Model bord1_5("l{1e-7}-r{47}l{100e-6}-r{47}c{1e-6}-r{47}");
	modelRange.clear();
	modelRange.push_back(eis::Range(5e-7, 5e-7, 1, true, 'l'));
	modelRange.push_back(eis::Range(1e2, 1e4, SWEEP_DEPTH, true, 'r'));
	modelRange.push_back(eis::Range(1e-6, 1e-3, SWEEP_DEPTH, true, 'l'));
	modelRange.push_back(eis::Range(1e1, 1e3, SWEEP_DEPTH, true, 'r'));
	modelRange.push_back(eis::Range(1e-9, 1e-5, SWEEP_DEPTH, true, 'c'));
	modelRange.push_back(eis::Range(1e1, 1e3, SWEEP_DEPTH, true, 'r'));
	assert(bord1_5.checkParameterRanges(modelRange));
	dataset->addModel(bord1_5, modelRange);

	eis::Model bord2("l{1e-7}-r{1e3}c{1e-6}-r{1e3}c{1e-9}-r{47}");
	modelRange.clear();
	modelRange.push_back(eis::Range(5e-7, 5e-7, 1, true, 'l'));
	modelRange.push_back(eis::Range(1e2, 1e4, SWEEP_DEPTH, true, 'r'));
	modelRange.push_back(eis::Range(1e-12, 1e-5, SWEEP_DEPTH, true, 'c'));
	modelRange.push_back(eis::Range(1e1, 1e4, SWEEP_DEPTH, true, 'r'));
	modelRange.push_back(eis::Range(1e-12, 1e-5, SWEEP_DEPTH, true, 'c'));
	modelRange.push_back(eis::Range(1e1, 1e4, SWEEP_DEPTH, true, 'r'));
	assert(bord2.checkParameterRanges(modelRange));
	dataset->addModel(bord2, modelRange);

//	eis::Model bord4("l{1e-7}-r{100}-r{470}c{100e-12}-r{1e3}c{1e-9}-r{100}c{1e-6}");
//	modelRange.clear();
//	modelRange.push_back(eis::Range(5e-7, 5e-7, 1, true, 'l'));
//	modelRange.push_back(eis::Range(10, 1e4, SWEEP_DEPTH, true, 'r'));
//	modelRange.push_back(eis::Range(10, 1e4, SWEEP_DEPTH, true, 'r'));
//	modelRange.push_back(eis::Range(1e-12, 1e-5, SWEEP_DEPTH, true, 'c'));
//	modelRange.push_back(eis::Range(10, 1e4, SWEEP_DEPTH, true, 'r'));
//	modelRange.push_back(eis::Range(1e-12, 1e-5, SWEEP_DEPTH, true, 'c'));
//	modelRange.push_back(eis::Range(10, 1e4, SWEEP_DEPTH, true, 'r'));
//	modelRange.push_back(eis::Range(1e-12, 1e-5, SWEEP_DEPTH, true, 'c'));
//	assert(bord4.checkParameterRanges(modelRange));
//	dataset->addModel(bord4, modelRange);

	eis::Model bordTl("l{1e-7}-t{1e3, 1e-7, 5}");
	modelRange.clear();
	modelRange.push_back(eis::Range(5e-7, 5e-7, 1, true, 'l'));
	modelRange.push_back(eis::Range(1e2, 1e4, SWEEP_DEPTH*10, true, 't'));
	modelRange.push_back(eis::Range(1e-9, 1e-5, SWEEP_DEPTH*10, true, 't'));
	modelRange.push_back(eis::Range(5, 5, 1, true, 't'));
	assert(bordTl.checkParameterRanges(modelRange));
	dataset->addModel(bordTl, modelRange);

	Log(Log::DEBUG)<<__func__<<" with "<<dataset->size().value()<<" examples\n";
	return dataset;
}

EisGeneratorDataset* EisGeneratorDataset::getNeisGeneratedDatasetAlt(int64_t outputSize)
{
	EisGeneratorDataset* dataset = new EisGeneratorDataset(outputSize);

	std::vector<eis::Range> modelRange;

	eis::Model bord1_1("(c{1e-6}-r{47})(c{1e-6}-r{1e3})(c{1e-6}-r{22e3})");
	modelRange.push_back(eis::Range(1e-7, 1e-5, rand() % (SWEEP_DEPTH/2)+2, true, 'c'));
	modelRange.push_back(eis::Range(10, 100e3, rand() % (SWEEP_DEPTH/2)+2, true, 'r'));
	modelRange.push_back(eis::Range(1e-7, 1e-5, rand() % (SWEEP_DEPTH/2)+2, true, 'c'));
	modelRange.push_back(eis::Range(10, 100e3, rand() % (SWEEP_DEPTH/2)+2, true, 'r'));
	modelRange.push_back(eis::Range(1e-7, 1e-5, rand() % (SWEEP_DEPTH/2)+2, true, 'c'));
	modelRange.push_back(eis::Range(10, 100e3, rand() % (SWEEP_DEPTH/2)+2, true, 'r'));
	assert(bord1_1.checkParameterRanges(modelRange));
	dataset->addModel(bord1_1, modelRange);

	eis::Model bord1_3("r{0.5}-c{10e-6}r{0.5}");
	modelRange.clear();
	modelRange.push_back(eis::Range(0.1, 10, (SWEEP_DEPTH/2)+2, true, 'r'));
	modelRange.push_back(eis::Range(1e-6, 1e-4, (SWEEP_DEPTH/2)+2, true, 'c'));
	modelRange.push_back(eis::Range(0.1, 10, (SWEEP_DEPTH/2)+2, true, 'r'));
	assert(bord1_3.checkParameterRanges(modelRange));
	dataset->addModel(bord1_3, modelRange);

	eis::Model bord1_4("r{100}c{1e-6}-r{100}c{100e-9}-r{100}c{100e-12}");
	modelRange.clear();
	modelRange.push_back(eis::Range(1e1, 1e4, rand() % (SWEEP_DEPTH/2)+2, true, 'r'));
	modelRange.push_back(eis::Range(1e-12, 1e-6, rand() % (SWEEP_DEPTH/2)+2, true, 'c'));
	modelRange.push_back(eis::Range(1e1, 1e4, rand() % (SWEEP_DEPTH/2)+2, true, 'r'));
	modelRange.push_back(eis::Range(1e-12, 1e-6, rand() % (SWEEP_DEPTH/2)+2, true, 'c'));
	modelRange.push_back(eis::Range(1e1, 1e4, rand() % (SWEEP_DEPTH/2)+2, true, 'r'));
	modelRange.push_back(eis::Range(1e-12, 1e-6, rand() % (SWEEP_DEPTH/2)+2, true, 'c'));
	assert(bord1_4.checkParameterRanges(modelRange));
	dataset->addModel(bord1_4, modelRange);

	eis::Model bord1_5("r{47}l{100e-6}-r{47}c{1e-6}-r{47}");
	modelRange.clear();
	modelRange.push_back(eis::Range(1e1, 1e3, rand() % (SWEEP_DEPTH/2)+2, true, 'r'));
	modelRange.push_back(eis::Range(1e-6, 1e-3, rand() % (SWEEP_DEPTH/2)+2, true, 'l'));
	modelRange.push_back(eis::Range(1e1, 1e3, rand() % (SWEEP_DEPTH/2)+2, true, 'r'));
	modelRange.push_back(eis::Range(1e-9, 1e-5, rand() % (SWEEP_DEPTH/2)+2, true, 'c'));
	modelRange.push_back(eis::Range(1e1, 1e3, rand() % (SWEEP_DEPTH/2)+2, true, 'r'));
	assert(bord1_5.checkParameterRanges(modelRange));
	dataset->addModel(bord1_5, modelRange);

	eis::Model bord2("r{1e3}c{1e-6}-r{1e3}c{1e-9}-r{47}");
	modelRange.clear();
	modelRange.push_back(eis::Range(1e1, 1e4, rand() % (SWEEP_DEPTH/2)+2, true, 'r'));
	modelRange.push_back(eis::Range(1e-12, 1e-5, rand() % (SWEEP_DEPTH/2)+2, true, 'c'));
	modelRange.push_back(eis::Range(1e1, 1e4, rand() % (SWEEP_DEPTH/2)+2, true, 'r'));
	modelRange.push_back(eis::Range(1e-12, 1e-5, rand() % (SWEEP_DEPTH/2)+2, true, 'c'));
	modelRange.push_back(eis::Range(1e1, 1e4, rand() % (SWEEP_DEPTH/2)+2, true, 'r'));
	assert(bord2.checkParameterRanges(modelRange));
	dataset->addModel(bord2, modelRange);

//	eis::Model bord4("r{100}-r{470}c{100e-12}-r{1e3}c{1e-9}-r{100}c{1e-6}");
//	modelRange.clear();
//	modelRange.push_back(eis::Range(10, 1e4, rand() % (SWEEP_DEPTH/2)+2, true, 'r'));
//	modelRange.push_back(eis::Range(10, 1e4, rand() % (SWEEP_DEPTH/2)+2, true, 'r'));
//	modelRange.push_back(eis::Range(1e-12, 1e-5, rand() % (SWEEP_DEPTH/2)+2, true, 'c'));
//	modelRange.push_back(eis::Range(10, 1e4, rand() % (SWEEP_DEPTH/2)+2, true, 'r'));
//	modelRange.push_back(eis::Range(1e-12, 1e-5, rand() % (SWEEP_DEPTH/2)+2, true, 'c'));
//	modelRange.push_back(eis::Range(10, 1e4, rand() % (SWEEP_DEPTH/2)+2, true, 'r'));
//	modelRange.push_back(eis::Range(1e-12, 1e-5, rand() % (SWEEP_DEPTH/2)+2, true, 'c'));
//	assert(bord4.checkParameterRanges(modelRange));
//	dataset->addModel(bord4, modelRange);

	eis::Model bordTl("t{1e3, 1e-7, 3}");
	modelRange.clear();
	modelRange.push_back(eis::Range(1e2, 1e4, SWEEP_DEPTH, true, 't'));
	modelRange.push_back(eis::Range(1e-9, 1e-5, SWEEP_DEPTH, true, 't'));
	modelRange.push_back(eis::Range(3, 3, 1, true, 't'));
	assert(bordTl.checkParameterRanges(modelRange));
	dataset->addModel(bordTl, modelRange);

	Log(Log::DEBUG)<<__func__<<" with "<<dataset->size().value()<<" examples\n";
	return dataset;
}

std::string EisGeneratorDataset::getModelStrFromClass(size_t classNum)
{
	if(classNum >= models.size())
		return "invalid";

	else return models[classNum].getModelStr();
}
