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
	return "r-rc-rc";
}
