#include "trashgenerator.h"
#include <cstddef>
#include <vector>
#include <assert.h>
#include <cmath>

TrashDataset::TrashDataset(int64_t outputSizeIn, int64_t classCountIn): outputSize(outputSizeIn), classCount(classCountIn)
{

}

torch::data::Example<torch::Tensor, torch::Tensor> TrashDataset::get(size_t index)
{
	std::default_random_engine rande(index);
	torch::TensorOptions options;
	options = options.dtype(torch::kFloat32);
	options = options.layout(torch::kStrided);
	options = options.device(torch::kCPU);
	torch::Tensor input = torch::zeros({static_cast<long int>(outputSize)}, options);

	size_t classNum = rande() % classCount;

	float* dataPtr = input.contiguous().data_ptr<float>();
	dataPtr[0] = (rande() % 1000)/1000.0f;
	size_t sectionSize = (rande() % (outputSize/6)) + outputSize/6;
	float sloap = (static_cast<int>(rande() % 200)-100)/10000.0f;
	for(int64_t i = 1; i < outputSize; ++i)
	{
		dataPtr[i] = dataPtr[i-1] + sloap;
		if(dataPtr[i] > 1)
			dataPtr[i] = 1;
		else if(dataPtr[i] < -1)
			dataPtr[i] = -1;
		if(sectionSize-- == 0)
		{
			sectionSize = (rande() % (outputSize/6)) + outputSize/6;
			sloap = (static_cast<int>(rande() % 800)-400)/10000.0f;
		}
	}

	options = options.dtype(torch::kInt64);
	torch::Tensor output = torch::empty({1}, options);
	output[0] = static_cast<float>(classNum);

	return torch::data::Example<torch::Tensor, torch::Tensor>(input, output);
}

c10::optional<size_t> TrashDataset::size() const
{
	return 1e6;
}
