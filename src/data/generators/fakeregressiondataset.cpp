#include "fakeregressiondataset.h"
#include <ATen/ops/linspace.h>
#include <ATen/ops/rand.h>

#include "tensoroptions.h"

FakeRegressionDatasetGenerator::FakeRegressionDatasetGenerator(int64_t outputSizeIn):
outputSize(outputSizeIn)
{
}

bool FakeRegressionDatasetGenerator::isMulticlass()
{
	return true;
}

torch::data::Example<torch::Tensor, torch::Tensor> FakeRegressionDatasetGenerator::getImpl(size_t index)
{
	torch::Tensor parameters = torch::rand({5}, tensorOptCpu<float>(false))*1000-500;
	/*parameters[0] = 100;
	parameters[1] = -370;
	parameters[2] = 41;
	parameters[3] = 358;
	parameters[4] = 687;*/
	torch::Tensor x = torch::linspace(0, 1, outputSize, tensorOptCpu<float>(false));
	torch::Tensor out = parameters[0]*x + parameters[1]*torch::pow(x, 2) +
		parameters[2]*torch::pow(x, 3) + parameters[3]*torch::pow(x, 4)+ parameters[4];
	return torch::data::Example<torch::Tensor, torch::Tensor>(out, parameters);
}

size_t FakeRegressionDatasetGenerator::classesCount() const
{
	return 5;
}

c10::optional<size_t> FakeRegressionDatasetGenerator::size() const
{
	return 1e5;
}

size_t FakeRegressionDatasetGenerator::classForIndex(size_t index)
{
	(void)index;
	return 0;
}

std::string FakeRegressionDatasetGenerator::modelStringForClass(size_t classNum)
{
	(void)classNum;

	switch (classNum)
	{
		case 0:
			return "r-0";
		case 1:
			return "r-1";
		case 2:
			return "c-2";
		case 3:
			return "r-3";
		case 4:
			return "c-4";
		default:
			return "r-rc-rc";
	}
}

c10::optional<torch::Tensor> FakeRegressionDatasetGenerator::freqRange()
{
	torch::Tensor out = torch::zeros({3}, tensorOptCpu<fvalue>(false));
	out[0] = 0;
	out[1] = 1;
	out[2] = static_cast<double>(outputSize);
	return out;
}
