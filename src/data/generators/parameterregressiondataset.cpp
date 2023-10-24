#include <eisdrt/eistorchdrt.h>
#include <eisdrt/types.h>

#include "parameterregressiondataset.h"
#include "data/eistotorch.h"

ParameterRegressionDatasetGenerator::ParameterRegressionDatasetGenerator(const std::string& modelStr, int64_t outputSize,
																		 double noiseI, bool drtI, size_t targetSize):
model(modelStr), omega(10, 1e6, drtI ? outputSize : outputSize/2, true), noise(noiseI), drt(drtI)
{
	sweepCount = model.getRequiredStepsForSweeps();
	if(sweepCount < 1e6 && targetSize == 0)
		model.setParamSweepCountClosestTotal(1e6);
	else if(sweepCount > 10e6 && targetSize == 0)
		model.setParamSweepCountClosestTotal(10e6);
	else if(targetSize != 0)
		model.setParamSweepCountClosestTotal(targetSize);
	sweepCount = model.getRequiredStepsForSweeps();
	parameterCount = model.getParameterCount();
	model.compile();
}

bool ParameterRegressionDatasetGenerator::isMulticlass()
{
	return true;
}

torch::data::Example<torch::Tensor, torch::Tensor> ParameterRegressionDatasetGenerator::getImpl(size_t index)
{
	std::vector<eis::DataPoint> data = model.executeSweep(omega, index);
	std::vector<eis::Componant*> componants = model.getFlatComponants();

	assert(data.size());
	torch::Tensor input;
	if(!drt)
	{
		input = eisToTorch(data);
	}
	else
	{
		FitMetics fm;
		try {
			input = calcDrtTorch(data, fm,  FitParameters(1000));
		}
		catch (const drt_errror& ex)
		{
			Log(Log::DEBUG)<<"Drt calculation failed! using next index";
			index++;
			if(index >= size().value())
				index = 0;
			return getImpl(index);
		}
	}

	torch::TensorOptions options;
	options = options.layout(torch::kStrided);
	options = options.device(torch::kCPU);
	options = options.dtype(torch::kFloat32);
	torch::Tensor output = torch::empty({static_cast<int64_t>(model.getParameterCount())}, options);

	size_t i = 0;
	auto tensorAccessor = output.accessor<float, 1>();
	for(eis::Componant * const componant : componants)
	{
		std::vector<eis::Range> ranges = componant->getParamRanges();
		for(const eis::Range& range : ranges)
		{
			tensorAccessor[i] = range.stepValue();
			++i;
		}
	}

	return torch::data::Example<torch::Tensor, torch::Tensor>(input, output);
}

size_t ParameterRegressionDatasetGenerator::classesCount() const
{
	return parameterCount;
}

c10::optional<size_t> ParameterRegressionDatasetGenerator::size() const
{
	return sweepCount;
}

size_t ParameterRegressionDatasetGenerator::classForIndex(size_t index)
{
	(void)index;
	return 0;
}

std::string ParameterRegressionDatasetGenerator::modelStringForClass(size_t classNum)
{
	std::vector<eis::Componant*> componants = model.getFlatComponants();
	for(eis::Componant* componant : componants)
	{
		if(classNum > componant->paramCount())
			classNum -= componant->paramCount();
		else
			return std::to_string(componant->getComponantChar()) + "p" + std::to_string(classNum);
	}
	return model.getModelStr();
}
