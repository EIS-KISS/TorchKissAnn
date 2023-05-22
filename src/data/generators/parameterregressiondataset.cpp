#include <eisdrt/eistorchdrt.h>

#include "parameterregressiondataset.h"
#include "data/eistotorch.h"

ParameterRegressionDataset::ParameterRegressionDataset(const std::string& modelStr, int64_t outputSize, double noiseI, bool drtI):
model(modelStr), omega(10, 1e6, drtI ? outputSize : outputSize/2, true), noise(noiseI), drt(drtI)
{
	sweepCount = model.getRequiredStepsForSweeps();
	if(sweepCount < 1e6)
		model.setParamSweepCountClosestTotal(1e6);
	else if(sweepCount > 10e6)
		model.setParamSweepCountClosestTotal(10e6);
	sweepCount = model.getRequiredStepsForSweeps();
	parameterCount = model.getParameterCount();
}

bool ParameterRegressionDataset::isMulticlass()
{
	return true;
}

torch::data::Example<torch::Tensor, torch::Tensor> ParameterRegressionDataset::getImpl(size_t index)
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
		input = calcDrtTorch(data, fm,  FitParameters(1000));
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

size_t ParameterRegressionDataset::classesCount() const
{
	return parameterCount;
}

c10::optional<size_t> ParameterRegressionDataset::size() const
{
	return sweepCount;
}

size_t ParameterRegressionDataset::classForIndex(size_t index)
{
	(void)index;
	return 0;
}

std::string ParameterRegressionDataset::modelStringForClass(size_t classNum)
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
